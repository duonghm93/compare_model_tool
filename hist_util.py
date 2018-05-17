import pickle
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


def load_data(files, modify_func):
    result = {}
    for file in files:
        filename = os.path.basename(file)
        hist = pickle.load(open(file, 'rb')) 
        filename_part = filename.split('.')[0].split('_')
        spid = filename_part[0]
        data_type = filename_part[-1]
        field_name = '_'.join(filename_part[1:-1])
        if result.get(spid, None) is None:
            result[spid] = {}
        if result[spid].get(data_type, None) is None:
            result[spid][data_type] = {}
        result[spid][data_type][field_name] = modify_func(hist)
    return result
	

def convert_norm_hist(hist):
    freqency = hist[1]
    total_size = sum(freqency)
    norm_fq = [count/total_size for count in freqency]
    return (hist[0], norm_fq)
	

def get_hist_bar_plot(hist, width=None):
    if width is None:
        width = hist[0][1]-hist[0][0]
    return plt.bar(hist[0][:-1], hist[1], width=width)
	

def get_hist(hist_dic, spid, data_type, field):
    return hist_dic[str(spid)][data_type][field]
	

def calc_pdf(hist):    
    fq = hist[1]
    total_size = sum(fq)
    pdf = np.cumsum(fq)    
    pdf = [csum/total_size for csum in pdf]    
    return (hist[0], pdf)
	

def get_part_hist(hist, min_bin_value, max_bin_value):
    def _get_first_larger_pos(lst, val):
        return next(x[0] for x in enumerate(lst) if x[1] >= val)
    def _get_last_smaller_pos(lst, val):
        return len(lst) - next(x[0] for x in enumerate(reversed(lst)) if x[1] <= val)
    bins, fq = hist[0], hist[1]
    first_bin_pos = _get_first_larger_pos(bins, min_bin_value)
    last_bin_pos = _get_last_smaller_pos(bins, max_bin_value)
    part_bins = bins[first_bin_pos:last_bin_pos]
    part_fq = fq[first_bin_pos:last_bin_pos-1]
    return (part_bins, part_fq)
	

# def check_field_by_bar(hist_dic, spid, field):
#     hist_click = get_hist(hist_dic, spid, 'isclick', field)
#     hist_nonclick = get_hist(hist_dic, spid, 'nonclick', field)
#     plt.figure(figsize=(15,5))
#     plt.title('%d_%s'%(spid,field))
#     bar_click = plt.bar(hist_click[0][:-1], hist_click[1], width=0.01)        
#     bar_nonclick = plt.bar(hist_nonclick[0][:-1], hist_nonclick[1], width=0.01)
#     plt.legend([bar_click, bar_nonclick], ['click', 'nonclick'])
#     plt.show()
#     plt.close()
	
	
def get_merge_fq(hist, min_bin, max_bin):
    part_hist = get_part_hist(hist, min_bin, max_bin)
    return sum(part_hist[1])

	
def get_chunks(l, chunk_size):
    return [l[i:i+chunk_size] for i in range(0,len(l),chunk_size)]
	

def get_merge_hist(hist, bin_size):
    bins = hist[0]
    old_bin_size = bins[1] - bins[0]
    if bin_size > old_bin_size:
        group_size = round(bin_size / old_bin_size)
        lst_chunks = get_chunks(bins[:-1], group_size)
        new_bins = [chunk[0] for chunk in lst_chunks]+[bins[-1]]
        new_fq = [get_merge_fq(hist, chunk[0], chunk[-1]) for chunk in lst_chunks]
        return (new_bins, new_fq)
    else:
        print('bin_size is too small')
        return hist
		
		
def check_field_by_line(hist_dic, spid, field, min_bin=0.0, max_bin=1.0, x_tick_num=20, merge_bin_size=None):     
    step = (max_bin-min_bin)/x_tick_num
    x_ticks = [min_bin+x*step for x in range(x_tick_num+1)]
    hist_click = get_hist(hist_dic, spid, 'isclick', field)
    hist_nonclick = get_hist(hist_dic, spid, 'nonclick', field)
    if merge_bin_size is not None:
        hist_click = get_merge_hist(hist_click, merge_bin_size)
        hist_nonclick = get_merge_hist(hist_nonclick, merge_bin_size)
    pdf_click = calc_pdf(hist_click)
    pdf_nonclick = calc_pdf(hist_nonclick)
    if min_bin != 0.0 or max_bin != 1.0:
        hist_click = get_part_hist(hist_click, min_bin, max_bin)
        hist_nonclick = get_part_hist(hist_nonclick, min_bin, max_bin)
        pdf_click = get_part_hist(pdf_click, min_bin, max_bin)
        pdf_nonclick = get_part_hist(pdf_nonclick, min_bin, max_bin)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,10)) 
    fig.suptitle('%d_%s'%(spid,field))
    axes[0].plot(hist_nonclick[0][:-1], hist_nonclick[1], label='nonclick', alpha=0.5)
    axes[0].plot(hist_click[0][:-1], hist_click[1], label='click', alpha=0.5)    
    axes[0].set_xticks(x_ticks)
    axes[0].grid()
    axes[0].legend()    
    axes[1].plot(pdf_nonclick[0][:-1], pdf_nonclick[1], label='nonclick')
    axes[1].plot(pdf_click[0][:-1], pdf_click[1], label='click')
    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks([x*0.1 for x in range(11)])
    axes[1].grid()
    axes[1].legend()
#     fig.savefig('%d_%s.png'%(spid,field), dpi=600)
    plt.show()    
    plt.close()
	

def check_field_by_bar(hist_dic, spid, field, min_bin=0.0, max_bin=1.0, x_tick_num=20, merge_bin_size=None):     
    step = (max_bin-min_bin)/x_tick_num
    x_ticks = [min_bin+x*step for x in range(x_tick_num+1)]
    hist_click = get_hist(hist_dic, spid, 'isclick', field)
    hist_nonclick = get_hist(hist_dic, spid, 'nonclick', field)
    if merge_bin_size is not None:
        hist_click = get_merge_hist(hist_click, merge_bin_size)
        hist_nonclick = get_merge_hist(hist_nonclick, merge_bin_size)
    pdf_click = calc_pdf(hist_click)
    pdf_nonclick = calc_pdf(hist_nonclick)
    if min_bin != 0.0 or max_bin != 1.0:
        hist_click = get_part_hist(hist_click, min_bin, max_bin)
        hist_nonclick = get_part_hist(hist_nonclick, min_bin, max_bin)
        pdf_click = get_part_hist(pdf_click, min_bin, max_bin)
        pdf_nonclick = get_part_hist(pdf_nonclick, min_bin, max_bin)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,10)) 
    fig.suptitle('%d_%s'%(spid,field))
    axes[0].bar(hist_nonclick[0][:-1], hist_nonclick[1], label='nonclick', alpha=0.5, width=merge_bin_size)
    axes[0].bar(hist_click[0][:-1], hist_click[1], label='click', alpha=0.5, width=merge_bin_size)    
    axes[0].set_xticks(x_ticks)
    axes[0].grid()
    axes[0].legend()    
    axes[1].plot(pdf_nonclick[0][:-1], pdf_nonclick[1], label='nonclick')
    axes[1].plot(pdf_click[0][:-1], pdf_click[1], label='click')
    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks([x*0.1 for x in range(11)])
    axes[1].grid()
    axes[1].legend()
#     fig.savefig('%d_%s.png'%(spid,field), dpi=600)
    plt.show()    
    plt.close()
	
	
def show_hist_by_line(hist, min_bin=0.0, max_bin=1.0, merge_bin_size=None, x_tick_num=20):
    step = (max_bin-min_bin)/x_tick_num
    x_ticks = [min_bin+x*step for x in range(x_tick_num+1)]
    fig, ax = plt.subplots(figsize=(15,5))
    ax.grid()
    ax.set_xticks(x_ticks)
    if merge_bin_size is not None:
        hist = get_merge_hist(hist, merge_bin_size)
    if min_bin != 0.0 or max_bin != 1.0:
        hist = get_part_hist(hist, min_bin, max_bin)
    ax.plot(hist[0][:-1], hist[1])    
    plt.show()
    plt.close()
	
	
def show_hist_by_bar(hist, min_bin=0.0, max_bin=1.0, merge_bin_size=None, x_tick_num=20):
    step = (max_bin-min_bin)/x_tick_num
    x_ticks = [min_bin+x*step for x in range(x_tick_num+1)]
    fig, ax = plt.subplots(figsize=(15,5))
    ax.grid()
    ax.set_xticks(x_ticks)
    if merge_bin_size is not None:
        hist = get_merge_hist(hist, merge_bin_size)
    if min_bin != 0.0 or max_bin != 1.0:
        hist = get_part_hist(hist, min_bin, max_bin)
    ax.bar(hist[0][:-1], hist[1], width=merge_bin_size)    
    plt.show()
    plt.close()