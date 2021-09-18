# https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data#Absolute_Band_Powers
# ref: https://alexandre.barachant.org/blog/2017/02/05/P300-with-muse.html
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
#parser = argparse.ArgumentParser(description='A program for p300 detection in continuous EEG signal')
#parser.add_argument('-f', '--file', type = str, default = 'museMonitor_845.csv', help = 'input csv file')
#args = parser.parse_args()

csv_path = input("Enter the csv filename (keep the csv file in the same folder or pass the absolute path):")

df = pd.read_csv(csv_path.strip())
print(df)

def p300_segment(a_tp10, b_tp10, g_tp10, d_tp10, t_tp10, window = 420, scaling_factor = 0.6):
    """
    takes an eeg signal (filtered)
    detects a sudden peak from a valley in a close proximity with a ratio range
    """
    plt.plot(a_tp10)
    plt.xlabel('time index')
    plt.ylabel('amp')
    plt.title('Alpha TP-10')
    plt.savefig("Alpha_TP-10.png", dpi = 400)
    plt.clf()
    #plt.show()
    
    plt.plot(b_tp10)
    plt.xlabel('time index')
    plt.ylabel('amp')
    plt.title('Beta TP-10')
    #plt.show()
    plt.clf()
    plt.savefig("Beta_TP-10.png", dpi = 400)

    plt.plot(g_tp10)
    plt.xlabel('time index')
    plt.ylabel('amp')
    plt.title('Gamma TP-10')
    plt.savefig("Gamma_TP-10.png", dpi = 400)
    plt.clf()
    #plt.show()
    
    plt.plot(d_tp10)
    plt.xlabel('time index')
    plt.ylabel('amp')
    plt.title('Delta TP-10')
    plt.savefig("Delta_TP-10.png", dpi = 400)
    plt.clf()
    #plt.show()
    
    plt.plot(t_tp10)
    plt.xlabel('time index')
    plt.ylabel('amp')
    plt.title('Theta TP-10')
    plt.savefig("Theta_TP-10.png", dpi = 400)
    plt.clf()
    #plt.show()
    
    sum_sig = a_tp10 + b_tp10 + g_tp10 + d_tp10 + t_tp10
    
    mean_base = np.mean(sum_sig)
    std_base = np.std(sum_sig) * scaling_factor
    

    

    
    # making epochs
    p300_locs = []
    p300_mins = []
    p300_maxs = []
    for epoch_s in range(len(sum_sig)):
        epoch = sum_sig[epoch_s:min(len(sum_sig), epoch_s + window)]
        seg_1 = epoch[:len(epoch)//2]
        seg_2 = epoch[len(epoch)//2:]
        # check for p300 conditions
        if np.max(seg_1) < mean_base and np.min(seg_1) < 1.7:
            if np.min(seg_2) > mean_base and np.max(seg_2) > (mean_base + std_base):
                # this is a p300
                p300_locs.append(epoch_s)
                p300_mins.append(np.min(seg_1))
                p300_maxs.append(np.max(seg_2))
                epoch_s += window
    print(p300_locs)
    
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
    plt.title('Avg TP-10 Signal')
    plt.savefig("Avg_TP-10_300.png", dpi = 400)
    plt.clf()
    #plt.show()
    
p300_segment(df["Alpha_TP10"], df["Beta_TP10"], df["Gamma_TP10"], df["Delta_TP10"], df["Theta_TP10"])