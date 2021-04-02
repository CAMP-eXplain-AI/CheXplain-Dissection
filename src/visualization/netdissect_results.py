import torch, os, matplotlib.pyplot as plt
from netdissect import nethook, imgviz, show, segmenter, renormalize, upsample, tally, pbar
from netdissect import setting
import torch.nn as nn
from typing import Any
from PIL import Image
import IPython
import sys
sys.path.append('/content/project/mlmi')
import torch
import numpy as np
from torchvision import models
from src.data.dataloader import get_segmentation_dataloader
from src.data.dataloader import get_nih_segmented_dataloaders
from netdissect import setting

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class CovidNetDissectResults():
    def __init__(self, model, dataset, dataset_path, model_layer, seglabels = None, segcatlabels = None):
        model = nethook.InstrumentedModel(model)
        model.cuda()
        model.eval()
        self.model = model
        self.layername = model_layer
        self.model.retain_layer(self.layername)

        self.topk = None
        self.unit_images = None
        self.iou99 = None

        self.upfn = upsample.upsampler(
            target_shape=(56, 56),
            data_shape=(7, 7),
        )

        if dataset == 'covid_seg':
            self.seglabels = ['No class', 'Left Lung', 'Right Lung', 'Cardiomediastinum', 'Airways',
                         'Ground Glass Opacities',
                         'Consolidation', 'Pleural Effusion', 'Pneumothorax', 'Endotracheal Tube',
                         'Central Venous Line',
                         'Monitoring Probes', 'Nosogastric Tube', 'Chest tube', 'Tubings']
            self.segcatlabels = [('No class', 'No class'), ('Left Lung', 'Left Lung'), ('Right Lung', 'Right Lung'),
                        ('Cardiomediastinum', 'Cardiomediastinum'), ('Airways', 'Airways'),
                        ('Ground Glass Opacities', 'Ground Glass Opacities'), ('Consolidation', 'Consolidation'),
                        ('Pleural Effusion', 'Pleural Effusion'), ('Pneumothorax', 'Pneumothorax'),
                        ('Endotracheal Tube', 'Endotracheal Tube'), ('Central Venous Line', 'Central Venous Line'),
                        ('Monitoring Probes', 'Monitoring Probes'), ('Nosogastric Tube', 'Nosogastric Tube'),
                        ('Chest tube', 'Chest tube'), ('Tubings', 'Tubings')]
            config = {
                'batch_size': 1,
                'input_size': (224, 224),
            }

            # Creating the dataloaders
            self.ds_loader = get_segmentation_dataloader(dataset_path, **config)
            self.ds = self.ds_loader.dataset
            # Specify the sample size in case of bigger dataset. Default is 100 for covid seg
            self.sample_size = 100
    
        self.rq = self._get_rq_vals()
        self.iv = imgviz.ImageVisualizer(224, source=self.ds, percent_level=0.99, quantiles=self.rq)

    def change_the_retained_layer(self, layername):
        self.layername = layername
        self.model.retain_layer(self.layername)
        self.topk = None
        self.unit_images = None
        self.iou99 = None
        # Restart the imviz for the new layer
        self.rq = self._get_rq_vals()
        self.iv = imgviz.ImageVisualizer(224, source=self.ds, percent_level=0.99, quantiles=self.rq)

    def _flatten_activations(self, batch, *args):
        image_batch = batch.cuda()
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername)
        hacts = self.upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

    def _get_rq_vals(self):
        rq = tally.tally_quantile(
            self._flatten_activations,
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10)
        return rq


    def _max_activations(self, batch, *args):
        image_batch = batch.cuda()
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername).cpu()
        return acts.view(acts.shape[:2] + (-1,)).max(2)[0]

    def _mean_activations(self, batch, *args):
        image_batch = batch.cuda()
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername)
        return acts.view(acts.shape[:2] + (-1,)).mean(2)

    def compute_topk_imgs(self, mode = 'mean'):
        if mode == 'mean':
            self.topk = tally.tally_topk(
            self._mean_activations,
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10
            )
        else: # It can only be max if not mean
            self.topk = tally.tally_topk(
            self._max_activations,
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10
            )

    def _compute_activations(self, image_batch, label_batch):
            image_batch = image_batch.cuda()
            _ = self.model(image_batch)
            acts_batch = self.model.retained_layer(self.layername)
            return acts_batch

    def compute_top_unit_imgs(self, mode = 'mean', k = 5):
        if self.topk is None:
            self.compute_topk_imgs(mode)
        self.unit_images = self.iv.masked_images_for_topk(
            self._compute_activations,
            self.ds,
            self.topk,
            k=k,
            num_workers=10,
            pin_memory=True)

    def show_seg_results(self):
        if self.unit_images is None:
            self.compute_top_unit_imgs()
        level_at_99 = self.rq.quantiles(0.99).cuda()[None, :, None, None]
        sample_size = 20

        def compute_selected_segments(batch, *args):
            img, seg = batch
            #     show(iv.segmentation(seg))
            image_batch = img.cuda()
            seg_batch = seg.cuda()
            _ = self.model(image_batch)
            acts = self.model.retained_layer(self.layername)
            hacts = self.upfn(acts)
            iacts = (hacts > level_at_99).float()  # indicator where > 0.99 percentile.
            return tally.conditional_samples(iacts, seg_batch)

        condi99 = tally.tally_conditional_mean(
            compute_selected_segments,
            dataset=self.ds,
            sample_size=sample_size, loader=self.ds_loader, pass_with_lbl=True)

        self.iou99= tally.iou_from_conditional_indicator_mean(condi99)
        bolded_string = "\033[1m" + self.layername + "\033[0m"

        print(bolded_string)
        iou_unit_label_99 = sorted([(
            unit, concept.item(), self.seglabels[int(concept)], bestiou.item())
            for unit, (bestiou, concept) in enumerate(zip(*self.iou99.max(0)))],
            key=lambda x: -x[-1])
        for unit, concept, label, score in iou_unit_label_99[:20]:
            show(['unit %d; iou %g; label "%s"' % (unit, score, label),
                  [self.unit_images[unit]]])

    def show_top_activating_imgs_per_units_with_seg(self, units, top_num = 1):
        if self.topk is None:
            self.compute_topk_imgs()
        top_indexes = self.topk.result()[1]
        show([
            ['unit %d' % u,
             'img %d' % i,
             'pred: %s' % [self.model(self.ds[i][0][None].cuda())],
             [self.iv.masked_image(
                 self.ds[i][0],
                 self.model.retained_layer(self.layername)[0],
                 u)], [self.iv.heatmap(self.model.retained_layer(self.layername)[0], u)], [self.iv.segmentation(self.ds[i][1])]
             ]
            for u in units
            for i in top_indexes[u, :top_num]
        ])

    def show_seg_gt(self, num_samples = 5):
        imgs = []
        seg = []
        for i in range(num_samples):
            img, lbl = self.ds[i]
            imgs.append(img)
            seg.append(lbl)
        show([(self.iv.image(imgs[i]), self.iv.segmentation(seg[i]),
               self.iv.segment_key_with_lbls(seg[i], self.seglabels))
              for i in range(len(seg))])

    def show_unique_concepts_graph(self, thresh = 0.04,  print_nums = False):
        if self.iou99 is None:
            self.show_seg_results()
        iou_threshold = thresh
        unit_label_99 = [
            (concept.item(), self.seglabels[concept],
             self.segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*self.iou99.max(0))]
        labelcat_list = [labelcat
                         for concept, label, labelcat, iou in unit_label_99
                         if iou > iou_threshold]
        return setting.graph_conceptcatlist(labelcat_list, cats=self.seglabels, print_nums = print_nums)


class NIHNetDissectResults():
    def __init__(self, model, dataset, dataset_path, model_layer, seglabels = None, segcatlabels = None, model_nm = None):
        model = nethook.InstrumentedModel(model)
        model.cuda()
        model.eval()
        self.model = model
        self.layername = model_layer
        self.model.retain_layer(self.layername)
        self.model_name = model_nm

        self.topk = None
        self.unit_images = None
        self.iou99 = None

        self.upfn = upsample.upsampler(
            target_shape=(56, 56),
            data_shape=(7, 7),
        )

        if dataset == 'nih_seg':
            if seglabels is not None:
                self.seglabels = seglabels
            else:
                self.seglabels = ['No Class', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule',
                                  'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
                                  'Fibrosis', 'Pleural_Thickening', 'Hernia']
            if segcatlabels is not None:
                self.segcatlabels = segcatlabels
            else:
                self.segcatlabels = [('No Class', 'No Class'), ('Atelectasis','Atelectasis'), ('Cardiomegaly','Cardiomegaly'),
                                     ('Effusion','Effusion'), ('Infiltrate','Infiltrate'),
                                     ('Mass','Mass'), ('Nodule','Nodule'), ('Pneumonia','Pneumonia'),
                                     ('Pneumothorax','Pneumothorax'), ('Consolidation','Consolidation'),
                                     ('Edema','Edema'), ('Emphysema','Emphysema'), ('Fibrosis','Fibrosis'),
                                     ('Pleural_Thickening','Pleural_Thickening'), ('Hernia','Hernia')]

            if model_nm == 'chexpert_noweights':
                batch_sz = 10
            else:
                batch_sz = 20
            
            config = {
                'batch_size': batch_sz,
                'input_size': (224, 224)
            }

            # Creating the dataloaders
            _, _, self.ds_loader = get_nih_segmented_dataloaders(dataset_path, **config)
            self.ds = self.ds_loader.dataset
            # Setting sample size
            self.sample_size = 100
    
        self.rq = self._get_rq_vals()
        self.iv = imgviz.ImageVisualizer(224, source=self.ds, percent_level=0.99, quantiles=self.rq)

    def change_the_retained_layer(self, layername):
        self.layername = layername
        self.model.retain_layer(self.layername)
        self.topk = None
        self.unit_images = None
        self.iou99 = None
        # Restart the imviz for the new layer
        self.rq = self._get_rq_vals()
        self.iv = imgviz.ImageVisualizer(224, source=self.ds, percent_level=0.99, quantiles=self.rq)

    def _flatten_activations(self, batch, *args):
        image_batch = batch.cuda()
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername)
        hacts = self.upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

    def _get_rq_vals(self):
        rq = tally.tally_quantile(
            self._flatten_activations,
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10)
        return rq

    def _max_activations(self, batch, *args):
        image_batch = batch.cuda()
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername).cpu()
        return acts.view(acts.shape[:2] + (-1,)).max(2)[0]

    def _mean_activations(self, batch, *args):
        image_batch = batch.cuda()
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername)
        return acts.view(acts.shape[:2] + (-1,)).mean(2)

    def compute_topk_imgs(self, mode = 'mean'):
        if mode == 'mean':
            self.topk = tally.tally_topk(
            self._mean_activations,
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10
            )
        else: # It can only be max if not mean
            self.topk = tally.tally_topk(
            self._max_activations,
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10
            )

    def _compute_activations(self, image_batch, label_batch):
            image_batch = image_batch.cuda()
            _ = self.model(image_batch)
            acts_batch = self.model.retained_layer(self.layername)
            return acts_batch

    def compute_top_unit_imgs(self, mode = 'mean', k = 5):
        if self.topk is None:
            self.compute_topk_imgs(mode)
        self.unit_images = self.iv.masked_images_for_topk(
            self._compute_activations,
            self.ds,
            self.topk,
            k=k,
            num_workers=10,
            pin_memory=True)

    def show_seg_results(self):
        if self.unit_images is None:
            self.compute_top_unit_imgs()
        level_at_99 = self.rq.quantiles(0.99).cuda()[None, :, None, None]
        sample_size = 20

        def compute_selected_segments(batch, *args):
            img, seg = batch
            #     show(iv.segmentation(seg))
            image_batch = img.cuda()
            seg_batch = seg.cuda()
            _ = self.model(image_batch)
            acts = self.model.retained_layer(self.layername)
            hacts = self.upfn(acts)
            iacts = (hacts > level_at_99).float()  # indicator where > 0.99 percentile.
            return tally.conditional_samples(iacts, seg_batch)

        condi99 = tally.tally_conditional_mean(
            compute_selected_segments,
            dataset=self.ds,
            sample_size=sample_size, loader=self.ds_loader, pass_with_lbl=True)

        self.iou99= tally.iou_from_conditional_indicator_mean(condi99)
        bolded_string = "\033[1m" + self.layername + "\033[0m"

        print(bolded_string)
        iou_unit_label_99 = sorted([(
            unit, concept.item(), self.seglabels[int(concept)], bestiou.item())
            for unit, (bestiou, concept) in enumerate(zip(*self.iou99.max(0)))],
            key=lambda x: -x[-1])
        for unit, concept, label, score in iou_unit_label_99[:20]:
            show(['unit %d; iou %g; label "%s"' % (unit, score, label),
                  [self.unit_images[unit]]])

    def show_top_activating_imgs_per_units_with_seg(self, units, top_num = 1):
        if self.topk is None:
            self.compute_topk_imgs()
        top_indexes = self.topk.result()[1]
        show([
            ['unit %d' % u,
             'img %d' % i,
             'pred: %s' % self._get_pred(i),
             [self.iv.masked_image(
                 self.ds[i.item()][0],
                 self.model.retained_layer(self.layername)[0],
                 u)], [self.iv.heatmap(self.model.retained_layer(self.layername)[0], u)], [self.iv.segmentation(self.ds[i.item()][1])]
             ]
            for u in units
            for i in top_indexes[u, :top_num]
        ])
    
    def _get_pred(self, i):
        pred = self.model(self.ds[i.item()][0][None].cuda())
        if self.model_name != 'brixia':
            pred_indx = torch.where(pred[0] == max(pred[0]))[0][0]
            pred_label = self.seglabels[pred_indx + 1]
            return pred_label
        return pred

    def show_seg_gt(self, num_samples = 5):
        imgs = []
        seg = []
        for i in range(num_samples):
            img, lbl = self.ds[i]
            imgs.append(img)
            seg.append(lbl)
        show([(self.iv.image(imgs[i]), self.iv.segmentation(seg[i]),
               self.iv.segment_key_with_lbls(seg[i], self.seglabels))
              for i in range(len(seg))])

    def show_unique_concepts_graph(self, thresh = 0.04):
        if self.iou99 is None:
            self.show_seg_results()
        iou_threshold = thresh
        unit_label_99 = [
            (concept.item(), self.seglabels[concept],
             self.segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*self.iou99.max(0))]
        labelcat_list = [labelcat
                         for concept, label, labelcat, iou in unit_label_99
                         if iou > iou_threshold]
        return setting.graph_conceptcatlist(labelcat_list, cats=self.seglabels)