from ..functions.profile import dwell, gcam0_d_v, gcam0_dv_p
from ..functions.misc import concat
from ..functions.plot import plot_cam_profile

class Trapezoid_cr_only:    
    dca = 0.5
    start_dwell_ca = 10
    end_dwell_ca = 10
    
    # segment 1, dwell
    seg1_ca2 = -48
    # segment 2, gcam0 d-v
    seg2_s2 = 0.5
    seg2_vr = 0.035
    seg2_amatch = 0.0385
    seg2_amx = 0.0385
    seg2_jmx = 0.0072476
    # segment 3, gcam0 dv-p
    seg3_s2 = 3
    seg3_vr = 0.035
    seg3_amx = 0.0175
    seg3_dmx = 0.0057
    seg3_jmx = 0.0072476
    # segment 4, gcam0 dv-p
    seg4_s2 = 0.7
    seg4_vr = 0.0525
    seg4_amx = 0.0175
    seg4_dmx = 0.0057
    seg4_jmx = 0.0072476
    # segment 5, gcam0 d-v
    seg5_vr = 0.0525
    seg5_amx = 0.055
    seg5_jmx = 0.0072476
    
    segments = None
    profile = None
    
    def __init__(self,
                 seg1_ca2,
                 seg2_s2, seg2_vr, seg2_amatch, seg2_amx, seg2_jmx,
                 seg3_s2, seg3_vr, seg3_amx, seg3_dmx, seg3_jmx,
                 seg4_s2, seg4_vr, seg4_amx, seg4_dmx, seg4_jmx,
                 seg5_vr, seg5_amx, seg5_jmx):
        """Initialize instance variables from input parameters"""
        # segment 1
        self.seg1_ca2 = seg1_ca2
        
        # segment 2
        self.seg2_s2 = seg2_s2
        self.seg2_vr = seg2_vr
        self.seg2_amatch = seg2_amatch
        self.seg2_amx = seg2_amx
        self.seg2_jmx = seg2_jmx
        
        # segment 3
        self.seg3_s2 = seg3_s2
        self.seg3_vr = seg3_vr
        self.seg3_amx = seg3_amx
        self.seg3_dmx = seg3_dmx
        self.seg3_jmx = seg3_jmx
        
        # segment 4
        self.seg4_s2 = seg4_s2
        self.seg4_vr = seg4_vr
        self.seg4_amx = seg4_amx
        self.seg4_dmx = seg4_dmx
        self.seg4_jmx = seg4_jmx
        
        # segment 5
        self.seg5_vr = seg5_vr
        self.seg5_amx = seg5_amx
        self.seg5_jmx = seg5_jmx
        
    def profile_generate(self):
        seg1 = dwell(self.seg1_ca2 - self.start_dwell_ca, self.seg1_ca2, self.dca, 0) 
        seg2 = gcam0_d_v(seg1.iloc[-1]['ca'],
                         self.dca, seg1.iloc[-1]['s'], self.seg2_s2, self.seg2_vr, 
                         0, self.seg2_amatch, self.seg2_amx, 0, self.seg2_jmx)    
        seg3 = gcam0_dv_p(seg2.iloc[-1]['ca'], self.dca, seg2.iloc[-1]['s'], 
                          self.seg3_s2, self.seg3_vr, 0, seg2.iloc[-1]['a'],
                          self.seg3_amx, self.seg3_dmx, self.seg3_jmx)    
        seg4 = gcam0_dv_p(seg3.iloc[-1]['ca'], self.dca, seg3.iloc[-1]['s'],
                          self.seg4_s2, self.seg4_vr, seg3.iloc[-1]['v'], seg3.iloc[-1]['a'],
                          self.seg4_amx, self.seg4_dmx, self.seg4_jmx)    
        seg5 = gcam0_d_v(seg4.iloc[-1]['ca'], self.dca, seg4.iloc[-1]['s'],
                         0, self.seg5_vr, seg4.iloc[-1]['v'], seg4.iloc[-1]['a'],
                         self.seg5_amx, 0, self.seg5_jmx)    
        seg6 = dwell(seg5.iloc[-1]['ca'], seg5.iloc[-1]['ca'] + self.end_dwell_ca, self.dca, 0)    
        
        self.segments = [seg1, seg2, seg3, seg4, seg5, seg6]
        self.profile = concat(self.segments)
        return self.profile

    def segments_plot(self):
        return plot_cam_profile(self.segments, ['seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6'])
    
    def profile_plot(self, title="profile"):
        return plot_cam_profile([self.profile], [title]) 
    
    def peak_cr_ca(self):
        """
        Get the cam angle at peak lift.

        """
        if self.profile is None:
            self.profile_generate()
        
        # Find the index of maximum lift
        peak_idx = self.profile['s'].idxmax()
        
        # Return the cam angle at that index
        return self.profile.loc[peak_idx, 'ca']
