KEYPOINT_DICT = {
	"left_eye_center"      	  : (1,0),
	"right_eye_center"     	  : (3,2),
	"left_eye_inner"       	  : (5,4),
	"left_eye_outer"       	  : (7,6),
	"right_eye_inner"      	  : (9,8),
	"right_eye_outer"      	  : (11,10),
	"left_eyebrow_inner"   	  : (13,12),
	"left_eyebrow_outer"   	  : (15,14),
	"right_eyebrow_inner"  	  : (17,16),
	"right_eyebrow_outer"  	  : (19,18),
	"nose_tip"             	  : (21,20),
	"left_mouth_corner"    	  : (23,22),
	"right_mouth_corner"   	  : (25,24),
	"center_mouth_top_lip" 	  : (27,26),
	"center_mouth_bottom_lip" : (29,28)
}

from subprocess import Popen

cmds = ["python new_method.py {}".format(name) for name in KEYPOINT_DICT]

processes = [Popen(cmd, shell=True) for cmd in cmds]