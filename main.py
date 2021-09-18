"""
author: github.com/zabir-nabil
eeg p300 detection
"""
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data#Absolute_Band_Powers
# ref: https://alexandre.barachant.org/blog/2017/02/05/P300-with-muse.html

# parser = argparse.ArgumentParser(description='A program for p300 detection in continuous EEG signal')
# parser.add_argument('-f', '--file', type = str, default = 'museMonitor_845.csv', help = 'input csv file')
# args = parser.parse_args()

def visualize_plot(eeg_channel_keys, eeg_channels_data, visualize):

    for i, ck in enumerate(eeg_channel_keys):
        plt.plot(eeg_channels_data[i])
        plt.xlabel('time index')
        plt.ylabel('amp')
        plt.title(ck)
        plt.savefig(f"{ck.replace(' ', '_')}.png", dpi = 400)
        if visualize == "yes":
            plt.show()
        plt.clf()



def p300_segment(final_eeg_agg, df_timestamps = None, window = 420, scaling_factor = 0.6, div_factor = 1.21, visualize = False):
    """
    takes an eeg signal (filtered)
    detects a sudden peak from a valley in a close proximity with a ratio range
    """
    plt.plot(final_eeg_agg)
    plt.xlabel('time index')
    plt.ylabel('amp')
    plt.title('Aggregated EEG Data')
    plt.savefig("aggregated_eeg.png", dpi = 400)
    if visualize == "yes":
        plt.show()
    plt.clf()

    
    sum_sig = final_eeg_agg
    
    mean_base = np.mean(sum_sig)
    std_base = np.std(sum_sig) * scaling_factor
    
    print("signal stats:")
    print(mean_base)
    print(std_base)

    # making epochs
    p300_locs = []
    p300_mins = []
    p300_maxs = []
    for epoch_s in range(len(sum_sig)):
        epoch = sum_sig[epoch_s:min(len(sum_sig), epoch_s + window)]
        seg_1 = epoch[:len(epoch)//2]
        seg_2 = epoch[len(epoch)//2:]
        # check for p300 conditions
        if np.max(seg_1) < mean_base and np.min(seg_1) < mean_base / div_factor:
            if np.min(seg_2) > mean_base and np.max(seg_2) > (mean_base + std_base):
                # this is a p300
                p300_locs.append(epoch_s)
                p300_mins.append(np.min(seg_1))
                p300_maxs.append(np.max(seg_2))
                epoch_s += window
    

    # check if two adjacent p300 waves are too close or not
    
    if len(p300_locs) > 1:
        for i in range(1, len(p300_locs)):
            prev_p300 = p300_locs[i-1]
            cur_p300 = p300_locs[i]
            
            if cur_p300 - prev_p300 < window * 2:
                p300_locs[i-1] = p300_locs[i] # replace with the latter one, as there will be some delay in EVP
                p300_mins[i-1] = p300_mins[i]
                p300_maxs[i-1] = p300_maxs[i]
        
    
    plt.plot(sum_sig)
    for i in range(len(p300_locs)):
        plt.gca().add_patch(Rectangle((p300_locs[i],p300_mins[i]),window, p300_maxs[i] - p300_mins[i],
                        edgecolor='orange',
                        facecolor='none',
                        lw=4))
    plt.axhline(y=mean_base, color='r', linestyle='-')
    plt.xlabel('time index')
    plt.ylabel('amp')
    plt.title('P300 annotated')
    plt.savefig("p300_annotated.png", dpi = 400)
    if visualize == "yes":
        plt.show()
    plt.clf()

    p300_locs = list(set(p300_locs))
    print("detected p300 locations (refined):")
    print(p300_locs)
    print(f"window: {window}")
    if df_timestamps is not None:
        print("timestamps:")
        print([df_timestamps[i] for i in p300_locs])

    # result write
    timestamps = []
    sigs = []
    sig_ids = []
    sum_sig = list(sum_sig)
    if len(df_timestamps) != len(sum_sig):
        print("signal length mismatch")
    for i, pl in enumerate(p300_locs):
        timestamps.extend(  df_timestamps[ max(pl - window, 0) : min(pl + window, len(df_timestamps)) ]  )
        sigs.extend(  sum_sig[ max(pl - window, 0) : min(pl + window, len(df_timestamps)) ]  )
        n = len(sum_sig[ max(pl - window, 0) : min(pl + window, len(df_timestamps)) ])
        sig_ids.extend([f"p300_sig_id_{i}"]*n)

    results = pd.DataFrame(
        {
            "timestamps": timestamps,
            "agg_signal": sigs,
            "p300_signal_id": sig_ids
        }
    )

    try:
        print("saving results")
        results.to_csv("results.csv", index=False)
    except Exception as e:
        print(e)
        print("could not save the results due to path issues")

    



if __name__ == "__main__":
    # arguments parsing
    parser = OptionParser()
    parser.add_option('-i','--input_file', dest = 'input_file', default = "muse_data.csv",
                      help='input csv file path')
    parser.add_option('-c','--channels', dest = 'channels', default = "all_TP10",
                      help='comma separated channel names')
    parser.add_option('-s','--scaling_factor', dest = 'scaling_factor', default = 0.6,
                      help='scaling factor for signal comparing with mean')
    parser.add_option('-d','--div_factor', dest = 'div_factor', default = 1.21,
                      help='div factor for signal comparing with mean')
    parser.add_option('-e','--epoch', dest = 'epoch', default = 420,
                      help='number of samples to use for epoch')           
    parser.add_option('-v','--visualize', dest = 'visualize', default = "no",
                      help='Visualize the plots or save them')
    (options, args) = parser.parse_args()

    df = pd.read_csv(options.input_file)
    
    print("input dataset loaded")
    print(df.head())

    if options.channels == "all_TP10":
        try:
            eeg_chk = ["Alpha_TP10", "Beta_TP10", "Gamma_TP10", "Delta_TP10", "Theta_TP10"]
            eeg_channels = [df["Alpha_TP10"], df["Beta_TP10"], df["Gamma_TP10"], df["Delta_TP10"], df["Theta_TP10"]]
        except Exception as e:
            print("input channel data is inconsistent")
            print(e)
    else:
        channel_keys = [ck.strip() for ck in options.channels.split(",")]
        eeg_channels = []
        eeg_chk = []
        for ck in channel_keys:
            print(ck)
            if ck in df.columns:
                eeg_channels.append(df[ck])
                eeg_chk.append(ck)
            else:
                print(f"Channel {ck} is non-existent in the dataset.")
    print(eeg_chk)
    visualize_plot(eeg_chk, eeg_channels, options.visualize)

    if len(eeg_channels) > 1:
        final_eeg_sig = sum(eeg_channels) # sum as aggregation

    if "TimeStamp" in df.columns:
        df_timestamps = list(df["TimeStamp"])
    else:
        df_timestamps = None

    p300_segment(final_eeg_sig, df_timestamps = df_timestamps, window = options.epoch, scaling_factor = options.scaling_factor, 
                div_factor = options.div_factor, visualize = options.visualize)