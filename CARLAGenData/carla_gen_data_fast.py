'''
Generate CARLA Data
'''

import argparse
import logging
import math
import random
import time
import sys
import os
from os.path import join
import numpy as np

carla_path = 'D:/Downloads/CARLA_0.8.2'
sys.path.insert(0, join(carla_path, 'PythonClient'))

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

random.seed(0)

r2d = lambda x: x/math.pi*180

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        default=True,
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    # argparser.add_argument(
    #     '-i', '--images-to-disk',
    #     action='store_true',
    #     dest='save_images_to_disk',
    #     help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '--save_images_to_disk',
        default=True,
    )
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    
    argparser.add_argument('--x-res', type=int, default=2048)
    argparser.add_argument('--y-res', type=int, default=1024) 
    argparser.add_argument('--out-dir', type=str, default='D:/Data/Exp/CARLA_debug')
    argparser.add_argument('--n-episode', type=int, default=1400)
    argparser.add_argument('--n-frame', type=int, default=300)
    argparser.add_argument('--save-every-n-frames', type=int, default=1)
    argparser.add_argument('--cam-fov', type=float, default=50)
    argparser.add_argument('--cam-offset', nargs='+', type=float, default=[0.3, -0.10, 1.22], help='x y z')
    argparser.add_argument('--cam-rotation', nargs='+', type=float, default=[r2d(-0.025), r2d(0.02), 0], help='pitch yaw roll')


    return argparser.parse_args()

def run_carla_client(args):
    skip_frames = 150 # 100 # at 30 fps
    # number_of_episodes = args.n_episode
    # frames_per_episode = args.n_frame
    
    # weathers = list(range(number_of_episodes))
    # # random.shuffle(weathers)
    # weathers = [w % 14 + 1 for w in weathers]
    # https://carla.readthedocs.io/en/latest/carla_settings/

    n_weather = 14 # weathers starts from 1 
    n_player_start_spots = 152 # Town01
    # n_player_start_spots = 83 # Town02

    number_of_episodes = n_weather*n_player_start_spots
    frames_per_episode = args.n_frame

    start_spots, weathers = np.meshgrid(list(range(n_player_start_spots)), list(range(1, n_weather+1)))

    weathers = weathers.flatten()
    start_spots = start_spots.flatten()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    np.savetxt(join(args.out_dir, 'weathers.txt'), weathers, fmt='%d')
    np.savetxt(join(args.out_dir, 'start-spots.txt'), start_spots, fmt='%d')
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(number_of_episodes):
            # Start a new episode.

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=20,
                    NumberOfPedestrians=40,
                    WeatherId=weathers[episode],
                    # WeatherId=random.randrange(14) + 1,
                    # WeatherId=random.choice([1, 3, 7, 8, 14]),
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # The default camera captures RGB images of the scene.
                camera0 = Camera('RGB')
                # Set image resolution in pixels.
                camera0.set(FOV=args.cam_fov)
                camera0.set_image_size(args.x_res, args.y_res)
                # Set its position relative to the car in meters.
                camera0.set_position(*args.cam_offset)
                camera0.set_rotation(*args.cam_rotation)
                settings.add_sensor(camera0)

                # Let's add another camera producing ground-truth depth.
                camera1 = Camera('Depth', PostProcessing='Depth')
                camera1.set(FOV=args.cam_fov)
                camera1.set_image_size(args.x_res, args.y_res)
                camera1.set_position(*args.cam_offset)
                camera1.set_rotation(*args.cam_rotation)
                settings.add_sensor(camera1)

                camera2 = Camera('Seg', PostProcessing='SemanticSegmentation')
                camera2.set(FOV=args.cam_fov)
                camera2.set_image_size(args.x_res, args.y_res)
                camera2.set_position(*args.cam_offset)
                camera2.set_rotation(*args.cam_rotation)
                settings.add_sensor(camera2)

                if args.lidar:
                    lidar = Lidar('Lidar32')
                    lidar.set_position(0, 0, 2.50)
                    lidar.set_rotation(0, 0, 0)
                    lidar.set(
                        Channels=32,
                        Range=50,
                        PointsPerSecond=100000,
                        RotationFrequency=10,
                        UpperFovLimit=10,
                        LowerFovLimit=-30)
                    settings.add_sensor(lidar)

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            # number_of_player_starts = len(scene.player_start_spots)
            # player_start = random.randrange(number_of_player_starts)
            print(len(scene.player_start_spots))
            player_start = start_spots[episode]

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)

            frame = 0
            save_frame_idx = 0

            # extra_frames = 0
            # last_control = None
            # last_control_changed = 0
            # while frame < skip_frames + frames_per_episode + extra_frames:

            # Iterate every frame in the episode.
            for frame in range(skip_frames + frames_per_episode):
                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Print some of the measurements.
                print_measurements(measurements)

                # Save the images to disk if requested.
                if args.save_images_to_disk and frame >= skip_frames \
                    and (frame - skip_frames) % args.save_every_n_frames == 0:
                    # if last_control_changed < frame - args.save_every_n_frames:
                    #     extra_frames += args.save_every_n_frames
                    #     last_control_changed += args.save_every_n_frames
                    # else:
                    save_frame_idx += 1
                    for name, measurement in sensor_data.items():
                        filename = args.out_filename_format.format(episode+1, name, save_frame_idx)
                        measurement.save_to_disk(filename)

                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.

                if not args.autopilot:

                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)

                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.

                    control = measurements.player_measurements.autopilot_control
                    # if last_control:
                    #     for v1, v2 in zip(control.values(), last_control.values()):
                    #         if v1 != v2:
                    #             last_control_changed = frame
                    #             break

                    # control.steer += random.uniform(-0.1, 0.1)
                    client.send_control(control)

                # frame += 1

def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = join(args.out_dir, 'e{:0>6d}/{:s}/{:0>8d}')

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
