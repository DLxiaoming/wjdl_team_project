root@orangepi5plus:/# ros2 topic echo /tf_static --once
transforms:
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: head_pitch_link
  child_frame_id: head_camera_link
  transform:
    translation:
      x: 0.0755
      y: 0.1855
      z: -0.035
    rotation:
      x: 0.6975022537316997
      y: -0.11614116168115386
      z: 0.6975048158021092
      w: 0.11614158831106722
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: world
  child_frame_id: base_link
  transform:
    translation:
      x: 0.0
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: left_arm_link6
  child_frame_id: left_arm_flange
  transform:
    translation:
      x: 0.0
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: left_arm_flange
  child_frame_id: left_arm_end_link
  transform:
    translation:
      x: 0.1602
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: slide_link
  child_frame_id: left_arm_base_link
  transform:
    translation:
      x: 0.10174
      y: 0.02283
      z: 0.09475
    rotation:
      x: 0.27059805050674596
      y: 0.6532814834851055
      z: 0.653281481391271
      w: 0.2705980496394512
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: left_arm_flange
  child_frame_id: left_arm_eef_G2_base_link
  transform:
    translation:
      x: 0.009
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: base_link
  child_frame_id: left_wheel_link
  transform:
    translation:
      x: 0.0
      y: 0.16325
      z: 0.08401
    rotation:
      x: -0.4999999999966269
      y: -0.4999999999966269
      z: 0.5000018366025517
      w: -0.49999816339744835
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: right_arm_link6
  child_frame_id: right_arm_flange
  transform:
    translation:
      x: 0.0
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: right_arm_flange
  child_frame_id: right_arm_end_link
  transform:
    translation:
      x: 0.1602
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: slide_link
  child_frame_id: right_arm_base_link
  transform:
    translation:
      x: -0.10174
      y: 0.02283
      z: 0.09475
    rotation:
      x: -0.6532814834851055
      y: -0.27059805050674585
      z: 0.27059804963945105
      w: 0.653281481391271
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: right_arm_flange
  child_frame_id: right_arm_eef_G2_base_link
  transform:
    translation:
      x: 0.009
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
- header:
    stamp:
      sec: 1757241718
      nanosec: 692417941
    frame_id: base_link
  child_frame_id: right_wheel_link
  transform:
    translation:
      x: 0.0
      y: -0.16325
      z: 0.08401
    rotation:
      x: -0.4999999999966269
      y: -0.4999999999966269
      z: 0.5000018366025517
      w: -0.49999816339744835
---
root@orangepi5plus:/# 

