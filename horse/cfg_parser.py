"""
WARNING: spaghetti code.
"""

import numpy as np
import pickle
import os

def parser(model):
	"""
	Read the .cfg file to extract layers into `layers`
	as well as model-specific parameters into `meta`
	"""
	def _parse(l, i = 1):
		return l.split('=')[i].strip()

	with open(model, 'rb') as f:
		lines = f.readlines()

	lines = [line.decode() for line in lines]	
	
	meta = dict(); layers = list() # will contains layers' info
	h, w, c = [int()] * 3; layer = dict()
	for line in lines:
		line = line.strip()
		line = line.split('#')[0]
		if '[' in line:
			if layer != dict(): 
				if layer['type'] == '[net]': 
					h = layer['height']
					w = layer['width']
					c = layer['channels']
					meta['net'] = layer
				else:
					if layer['type'] == '[crop]':
						h = layer['crop_height']
						w = layer['crop_width']
					layers += [layer]				
			layer = {'type': line}
		else:
			try:
				i = float(_parse(line))
				if i == int(i): i = int(i)
				layer[line.split('=')[0].strip()] = i
			except:
				try:
					key = _parse(line, 0)
					val = _parse(line, 1)
					layer[key] = val
				except:
					'banana ninja yadayada'

	meta.update(layer) # last layer contains meta info
	if 'anchors' in meta:
		splits = meta['anchors'].split(',')
		anchors = [float(x.strip()) for x in splits]
		meta['anchors'] = anchors
	meta['model'] = model # path to cfg, not model name
	meta['inp_size'] = [h, w, c]
	return layers, meta

def cfg_yielder(model):
	"""
	yielding each layer information to initialize `layer`
	"""
	layers, meta = parser(model); yield meta;
	h, w, c = meta['inp_size']; l = w * h * c

	# Start yielding
	flat = False # flag for 1st dense layer
	conv = '.conv.' in model
	for i, d in enumerate(layers):
		#-----------------------------------------------------
		if d['type'] == '[crop]':
			yield ['crop', i]
		#-----------------------------------------------------
		elif d['type'] == '[convolutional]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size // 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			yield ['convolutional', i, size, c, n, 
				   stride, padding, batch_norm,
				   activation]
			if activation != 'linear': yield [activation, i]
			w_ = (w + 2 * padding - size) // stride + 1
			h_ = (h + 2 * padding - size) // stride + 1
			w, h, c = w_, h_, n
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[maxpool]':
			stride = d.get('stride', 1)
			size = d.get('size', stride)
			padding = d.get('padding', (size-1) // 2)
			yield ['maxpool', i, size, stride, padding]
			w_ = (w + 2*padding) // d['stride'] 
			h_ = (h + 2*padding) // d['stride']
			w, h = w_, h_
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[connected]':
			if not flat:
				yield ['flatten', i]
				flat = True
			activation = d.get('activation', 'logistic')
			yield ['connected', i, l, d['output'], activation]
			if activation != 'linear': yield [activation, i]
			l = d['output']
		#-----------------------------------------------------
		elif d['type'] == '[dropout]':
			pass
		else:
			exit('Layer {} not implemented'.format(d['type']))

		d['_size'] = list([h, w, c, l, flat])

	if not flat: meta['out_size'] = [h, w, c]
	else: meta['out_size'] = l