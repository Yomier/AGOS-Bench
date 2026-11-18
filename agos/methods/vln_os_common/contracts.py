import copy



def public_transition(result):
    return {key: copy.deepcopy(result[key]) for key in
            ('step', 'action', 'candidate_id', 'grid_cell', 'waypoint_id',
             'candidate_center_xy', 'pos', 'executed', 'error') if key in result}
