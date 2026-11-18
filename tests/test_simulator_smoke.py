"""Set AGOS_TEST_CARLA_PORT and AGOS_TEST_AIRSIM_PORT for a running local world."""
import os

import pytest


@pytest.mark.simulator
@pytest.mark.skipif(
    not (os.environ.get('AGOS_TEST_CARLA_PORT') and os.environ.get('AGOS_TEST_AIRSIM_PORT')),
    reason='set AGOS_TEST_CARLA_PORT and AGOS_TEST_AIRSIM_PORT to a running CarlaAir slot',
)
def test_native_clients_reach_the_running_world():
    import airsim
    import carla

    client = carla.Client('127.0.0.1', int(os.environ['AGOS_TEST_CARLA_PORT']))
    client.set_timeout(5.0)
    assert client.get_server_version()
    assert client.get_world().get_map().name
    drone = airsim.MultirotorClient(ip='127.0.0.1', port=int(os.environ['AGOS_TEST_AIRSIM_PORT']),
                                    timeout_value=5)
    try:
        assert drone.ping()
    finally:
        drone.client.close()
