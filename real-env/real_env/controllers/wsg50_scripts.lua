-- Nicolas Alt, 2014-09-04
-- Cheng Chi, 2023-07-27
-- Command-and-measure script
-- Tests showed about 30Hz rate
require "socket"
cmd.register(0xBA); -- Measure only
cmd.register(0xBB); -- Position PD

function hasbit(x, p)
  return x % (p + p) >= p       
end

function send_state()
    -- ==== Get measurements ====
    state = gripper.state();
    pos = mc.position();
    speed = mc.speed();
    force = mc.aforce();
    time = socket.gettime();

    if cmd.online() then
        -- Only the lowest byte of state is sent!
        cmd.send(id, etob(E_SUCCESS), state % 256, ntob(pos), ntob(speed), ntob(force), ntob(time));
    end
end

function process()
    id, payload = cmd.read();
    
    -- Position control
    if id == 0xBB then
        -- get args
        cmd_pos = bton({payload[1],payload[2],payload[3],payload[4]});
        cmd_vel = bton({payload[5],payload[6],payload[7],payload[8]});
        cmd_kp = bton({payload[9],payload[10],payload[11],payload[12]});
        cmd_kd = bton({payload[13],payload[14],payload[15],payload[16]});
        cmd_travel_force_limit = bton({payload[17],payload[18],payload[19],payload[20]});
        cmd_blocked_force_limit = bton({payload[21],payload[22],payload[23],payload[24]});

        -- get state
        pos = mc.position();
        vel = mc.speed();
        
        -- pd controller
        e = cmd_pos - pos;
        de = cmd_vel - vel;
        act_vel = cmd_kp * e + cmd_kd * de;

        printf("cmd_pos: %f, cmd_vel: %f, cmd_kp: %f, cmd_kd: %f, cmd_travel_force_limit: %f, cmd_blocked_force_limit: %f, pos: %f, vel: %f, act_vel: %f\n", cmd_pos, cmd_vel, cmd_kp, cmd_kd, cmd_travel_force_limit, cmd_blocked_force_limit, pos, vel, act_vel);
        
        -- command
        mc.speed(act_vel);
        
        -- force limit
        if mc.blocked() then
            mc.force(cmd_blocked_force_limit);
        else
            mc.force(cmd_travel_force_limit);
        end
    end
    
    --t_start = socket.gettime();
    send_state();
    --print(socket.gettime() - t_start);
    
end

while true do
    if cmd.online() then
        -- process()
        if not pcall(process) then
            print("Error occured")
            sleep(100)
        end
    else
        sleep(100)
    end
end
