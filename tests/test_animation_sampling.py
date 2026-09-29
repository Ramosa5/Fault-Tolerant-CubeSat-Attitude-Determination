import numpy as np
from src.visualization.animation import _animation_frame_indices, _downsample_indices


def test_animation_frame_cap_preserves_endpoints():
    t=np.arange(0.0,3000.1,0.1)
    idx=_animation_frame_indices(t,frame_stride=1,max_frames=600)
    assert len(idx) <= 601
    assert idx[0] == 0
    assert idx[-1] == len(t)-1


def test_animation_time_window_and_groundtrack_cap():
    t=np.arange(0.0,1000.1,0.1)
    idx=_animation_frame_indices(t,frame_stride=1,max_frames=200,start_time_s=250,end_time_s=450)
    assert t[idx[0]] >= 250
    assert t[idx[-1]] <= 450
    d=_downsample_indices(30000,max_points=900)
    assert len(d) <= 900
    assert d[0] == 0 and d[-1] == 29999
