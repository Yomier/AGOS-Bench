"""Bounded readiness probe for the two APIs and an actual RGB sensor frame."""
import sys
import threading
import time


def main(town, rpc, airsim_port):
    import airsim
    import carla
    from benchmark.native_runtime import require_client_guard
    require_client_guard()
    loaded = False
    asset = town + '_Opt' if town in {'Town03', 'Town04', 'Town05'} else town
    while True:  # Parent enforces a wall deadline even on blocked native RPCs.
        try:
            client = carla.Client('127.0.0.1', rpc)
            client.set_timeout(5.0)
            if not client.get_server_version():
                raise RuntimeError('no server version')
            world = client.get_world()
            if world.get_map().name.split('/')[-1].removesuffix('_Opt') != town:
                if not loaded:
                    loaded = True  # Mark before RPC; a timed out RPC may already have executed.
                    client.load_world(asset)
                raise RuntimeError('waiting for requested map')
            blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '320')
            blueprint.set_attribute('image_size_y', '240')
            loc = world.get_map().get_spawn_points()[0].location
            camera = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z+10)))
            frame = threading.Event()
            try:
                camera.listen(lambda image: frame.set() if len(image.raw_data) == 320*240*4 else None)
                if not frame.wait(4):
                    raise RuntimeError('RGB sensor did not deliver a frame')
            finally:
                try:
                    camera.stop()
                finally:
                    camera.destroy()
            drone = airsim.MultirotorClient(ip='127.0.0.1', port=airsim_port, timeout_value=5)
            try:
                if not drone.ping():
                    raise RuntimeError('no AirSim ping')
            finally:
                drone.client.close()
            return
        except Exception as exc:
            print(f'probe: {exc}', flush=True)
            time.sleep(1)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
