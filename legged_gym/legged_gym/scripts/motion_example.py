# -*- coding: UTF-8 -*-
import numpy

kDegree2Radian = 3.1415926 / 180

goal_angle_fl = numpy.zeros(3) 
goal_angle_hl = numpy.zeros(3) 
goal_angle_fr = numpy.zeros(3)
goal_angle_hr = numpy.zeros(3)
init_angle_fl = numpy.zeros(3)
init_angle_fr = numpy.zeros(3)
init_angle_hl = numpy.zeros(3)
init_angle_hr = numpy.zeros(3)

class MotionExample:
  def __init__(self):
    self.init_time = 0.0
  
  def CubicSpline(self, 
                  init_position, init_velocity,
                  goal_position, goal_velocity,
                  run_time, cycle_time, total_time):
    a=0.0
    b=0.0
    c = 0.0
    d = 0.0    
    d = init_position
    c = init_velocity
    a = (goal_velocity * total_time - 2 * goal_position + init_velocity * total_time + 
        2 * init_position) /pow(total_time, 3)
    b = (3 * goal_position - goal_velocity * total_time - 2 * init_velocity * total_time - 
        3 * init_position) / pow(total_time, 2)

    if run_time > total_time:
      run_time = total_time
    sub_goal_position = a * pow(run_time, 3) + b * pow(run_time, 2) + c * run_time + d

    if run_time + cycle_time > total_time:
      run_time = total_time - cycle_time
    sub_goal_position_next = a * pow(run_time + cycle_time, 3) + \
                             b * pow(run_time + cycle_time, 2) + \
                             c * (run_time + cycle_time) + d

    if run_time + cycle_time * 2 > total_time:
      run_time = total_time - cycle_time * 2
    sub_goal_position_next2 = a * pow(run_time + cycle_time * 2, 3) + \
                              b * pow(run_time + cycle_time * 2, 2) + \
                              c * (run_time + cycle_time * 2) + d
    return sub_goal_position, sub_goal_position_next, sub_goal_position_next2
  
  def SwingToAngle(self,
                   initial_angle, final_angle,
                   total_time, run_time,
                   cycle_time, side):
    goal_angle = numpy.zeros(3)
    goal_angle_next = numpy.zeros(3)
    goal_angle_next2 = numpy.zeros(3)
    goal_velocity = numpy.zeros(3)
    leg_side = 0

    if side == "FL":
      leg_side = 0
    elif side == "FR":
      leg_side = 1
    elif side == "HL":
      leg_side = 2
    elif side == "HR":
      leg_side = 3
    else:
      print("Leg Side Error!!!")

    final_angle = final_angle
    # kd = numpy.zeros(12, dtype=numpy.float32)
    target_leg = numpy.zeros(3, dtype=numpy.float32)
    for j in range(0, 3):
      goal_angle[j], goal_angle_next[j], goal_angle_next2[j] = \
        self.CubicSpline(initial_angle[j], 0, final_angle[j], 0, run_time,
                         cycle_time, total_time)
      goal_velocity[j] = (goal_angle_next[j] - goal_angle[j]) / cycle_time

    # kp[3 * leg_side] = 300
    # kp[3 * leg_side + 1] = 300
    # kp[3 * leg_side + 2] = 350

    # kd[3 * leg_side] = 2
    # kd[3 * leg_side + 1] = 2
    # kd[3 * leg_side + 2] = 2
    target_leg[0] = goal_angle[0]
    target_leg[1] = goal_angle[1]
    target_leg[2] = goal_angle[2]
    # target_dq[3 * leg_side] = goal_velocity[0]
    # target_dq[3 * leg_side + 1] = goal_velocity[1]
    # target_dq[3 * leg_side + 2] = goal_velocity[2]
    return target_leg
    


    
  def PreStandUp(self, target_q, target_dq, kp, kd, time):
    standup_time = 1.0
    cycle_time = 0.001
    goal_angle_fl = [0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian]
    goal_angle_fr = [0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian]
    goal_angle_hl = [0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian]
    goal_angle_hr = [0 * kDegree2Radian, -70 * kDegree2Radian, 150 * kDegree2Radian]

    if time <= self.init_time + standup_time:
      self.SwingToAngle(init_angle_fl, goal_angle_fl, standup_time, time - self.init_time,
                        cycle_time, "FL")
      self.SwingToAngle(init_angle_fl, goal_angle_fr, standup_time, time - self.init_time,
                        cycle_time, "FR")
      self.SwingToAngle(init_angle_fl, goal_angle_hl, standup_time, time - self.init_time,
                        cycle_time, "HL")
      self.SwingToAngle(init_angle_fl, goal_angle_hr, standup_time, time - self.init_time,
                        cycle_time, "HR")
   
  def StandUp(self, time):
    kp = numpy.zeros(12, dtype=numpy.float32)
    kd = numpy.zeros(12, dtype=numpy.float32)
    target_q = numpy.zeros(12, dtype=numpy.float32)
    standup_time = 1.5
    cycle_time = 0.001
    goal_angle_fl = [0 * kDegree2Radian, -42 * kDegree2Radian, 78 * kDegree2Radian] 
    goal_angle_fr = [0 * kDegree2Radian, -42 * kDegree2Radian, 78 * kDegree2Radian]
    goal_angle_hl = [0 * kDegree2Radian, -42 * kDegree2Radian, 78 * kDegree2Radian]
    goal_angle_hr = [0 * kDegree2Radian, -42 * kDegree2Radian, 78 * kDegree2Radian]

    if (time <= self.init_time + standup_time):
      target_q[0:3] = self.SwingToAngle(init_angle_fl, goal_angle_fl, standup_time, time - self.init_time,
                        cycle_time, "FL")
      target_q[3:6] = self.SwingToAngle(init_angle_fr, goal_angle_fr, standup_time, time - self.init_time,
                        cycle_time, "FR")
      target_q[6:9] = self.SwingToAngle(init_angle_hl, goal_angle_hl, standup_time, time - self.init_time,
                        cycle_time, "HL")
      target_q[9:12] = self.SwingToAngle(init_angle_hr, goal_angle_hr, standup_time, time - self.init_time,
                        cycle_time, "HR")
    else:
      for i in range(0, 12):
        # cmd.joint_cmd[i].torque = 0
        kp[i] = 300
        kd[i] = 4

      for i in range(0, 4):
        target_q[3*i]= 0
        target_q[3*i+1] = -42 * kDegree2Radian
        target_q[3*i+2] = 78 * kDegree2Radian
        # target_dq[3*i] = 0
        # target_dq[3*i+1] = 0
        # target_dq[3*i+2] = 0
    return target_q
        
  def GetInitData(self, q, time):
    self.init_time = time
    init_angle_fl[0] = q[0]
    init_angle_fl[1] = q[1]
    init_angle_fl[2] = q[2]

    init_angle_fr[0] = q[3]
    init_angle_fr[1] = q[4]
    init_angle_fr[2] = q[5]

    init_angle_hl[0] = q[6]
    init_angle_hl[1] = q[7]
    init_angle_hl[2] = q[8]

    init_angle_hr[0] = q[9]
    init_angle_hr[1] = q[10]
    init_angle_hr[2] = q[11]
