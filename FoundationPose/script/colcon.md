root@orangepi5plus:/# unzip MMK_URDF_0627.zip -d MMK_URDF_0627
bash: unzip: command not found
root@orangepi5plus:/# ls
bin                frames_2025-09-04_19.42.21.gv  mnt                 src
bin.usr-is-merged  frames_2025-09-04_21.05.45.gv  opt                 srv
boot               home                           proc                sys
core               lib                            root                tmp
dev                lib.usr-is-merged              run                 usr
etc                media                          sbin                var
fastdds.xml        MMK_URDF_0627.zip              sbin.usr-is-merged
root@orangepi5plus:/# cd src/
root@orangepi5plus:/src# ls
arm-models  mmk2_description
root@orangepi5plus:/src# cd ..
root@orangepi5plus:/# ls
bin                frames_2025-09-04_19.42.21.gv  mnt                 src
bin.usr-is-merged  frames_2025-09-04_21.05.45.gv  opt                 srv
boot               home                           proc                sys
core               lib                            root                tmp
dev                lib.usr-is-merged              run                 usr
etc                media                          sbin                var
fastdds.xml        MMK_URDF_0627.zip              sbin.usr-is-merged
root@orangepi5plus:/# cp -r src/* ~/ros2_ws/src/
root@orangepi5plus:/# cd ~
root@orangepi5plus:~# ls
ros2_ws
root@orangepi5plus:~# pwd
/root
root@orangepi5plus:~# cd ros2_ws/
root@orangepi5plus:~/ros2_ws# ls
src
root@orangepi5plus:~/ros2_ws# source /opt/ros/jazzy/setup.bash
root@orangepi5plus:~/ros2_ws# colcon build
[0.770s] WARNING:colcon.colcon_core.package_selection:Some selected packages are already built in one or more underlay workspaces:
	'airbot_description' is in: /opt/airbot_controller
If a package in a merged underlay workspace is overridden and it installs headers, then all packages in the overlay must sort their include directories by workspace order. Failure to do so may result in build failures or undefined behavior at run time.
If the overridden package is used by another package in any underlay, then the overriding package in the overlay must be API and ABI compatible or undefined behavior at run time may occur.

If you understand the risks and want to override a package anyways, add the following to the command line:
	--allow-overriding airbot_description

This may be promoted to an error in a future release of colcon-override-check.
Starting >>> airbot_description
Starting >>> mmk2_description
Finished <<< mmk2_description [4.20s]                                    
Finished <<< airbot_description [4.25s]

Summary: 2 packages finished [4.81s]
root@orangepi5plus:~/ros2_ws# source install/setup.bash
root@orangepi5plus:~/ros2_ws# ros2 pkg list | grep mmk2
mmk2_description
mmk2_drivers
mmk2_moveit_config
mmk2_ros2_interface
mmk2_state_manager
mmk2_teleop_tools
mmk2_types
root@orangepi5plus:~/ros2_ws# ros2 topic echo /tf_static | grep head_camera_link
