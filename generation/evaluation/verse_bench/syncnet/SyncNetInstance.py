#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from syncnet.SyncNetModel import *


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__()
        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda()
        self.__S__.eval()


    def evaluate2(self, frames_path, audio_path, fps=25.0, batch_size=20, vshift=15):
        flist = glob.glob(os.path.join(frames_path, '*.jpg'))
        flist.sort()
        images = []
        for fname in flist:
            images.append(cv2.resize(cv2.imread(fname), (224,224)))
        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))
        imtv = torch.Tensor(torch.from_numpy(im.astype(float)).float()).cuda()
        sample_rate, audio = wavfile.read(audio_path)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.Tensor(torch.from_numpy(cc.astype(float)).float()).cuda()
        if (float(len(audio)) / sample_rate) != (float(len(images)) / fps):
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different." % (
            float(len(audio)) / sample_rate, float(len(images)) / fps))
        min_length = min(len(images), math.floor(len(audio) / 640))
        lastframe = min_length - 5
        im_feat = []
        cc_feat = []
        tS = time.time()
        with torch.inference_mode():
            for i in range(0, lastframe, batch_size):
                im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
                im_in = torch.cat(im_batch, 0)
                im_out = self.__S__.forward_lip(im_in.cuda())
                im_feat.append(im_out)
                cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in
                            range(i, min(lastframe, i + batch_size))]
                cc_in = torch.cat(cc_batch, 0)
                cc_out = self.__S__.forward_aud(cc_in.cuda())
                cc_feat.append(cc_out)
            im_feat = torch.cat(im_feat, 0).cpu()
            cc_feat = torch.cat(cc_feat, 0).cpu()
            print('Compute time %.3f sec.' % (time.time() - tS))
            dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
            mdist = torch.mean(torch.stack(dists, 1), 1)
            minval, minidx = torch.min(mdist, 0)
            offset = vshift - minidx
            conf = torch.median(mdist) - minval
            # fdist = numpy.stack([dist[minidx].numpy() for dist in dists])
            # fconf = torch.median(mdist).numpy() - fdist
            # fconfm = signal.medfilt(fconf, kernel_size=9)
            numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset, minval, conf))
            dists_npy = numpy.array([dist.numpy() for dist in dists])
        return offset.numpy(), conf.numpy(), dists_npy

    def extract_feature(self, opt, videofile):
        self.__S__.eval()
        cap = cv2.VideoCapture(videofile)
        frame_num = 1
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break
            images.append(image)
        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        lastframe = len(images)-4
        im_feat = []
        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda())
            im_feat.append(im_out.data.cpu())
        im_feat = torch.cat(im_feat,0)
        print('Compute time %.3f sec.' % (time.time()-tS))
        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        self_state = self.__S__.state_dict()
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
