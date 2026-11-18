"""Evaluation-only receipts: tool fallback is never silently a valid formal score."""
from agos.agents.http_accounting import capture_http_attempts

MODEL_INFRA_STATUSES = {
    'vlm_timeout', 'vlm_http_error', 'vlm_auth_error', 'vlm_config_error',
    'vlm_empty_response',
}


def _append_event(state,event):
    if not hasattr(state,'paper_fault_events'):
        state.paper_fault_events=[]
    state.paper_fault_events.append(event)


def record_primary_infra(state,observation,action,parse_status=''):
    """Record a real INFRA_ERROR decision even when the next request recovers."""
    if getattr(action,'action',None) != 'INFRA_ERROR':
        return
    reason=str(getattr(action,'reason','') or '')
    status=str(parse_status or '')
    if status not in MODEL_INFRA_STATUSES:
        status=next((value for value in sorted(MODEL_INFRA_STATUSES) if value in reason),
                    status or 'vlm_infra_error')
    _append_event(state,{'step':observation.step,'role':observation.role,
        'component':'primary_model','status':status,'error':reason[:500],
        'kind':'model_transport','fallback':True})


def record_audit_failure(state,role,step,error):
    """Runner catch blocks call this instead of claiming an unobserved audit."""
    state.paper_audit_complete=False
    if not hasattr(state,'paper_audit_failures'):
        state.paper_audit_failures=[]
    failure={'role':role,'step':step,'error':f'{type(error).__name__}: {error}'[:500]}
    state.paper_audit_failures.append(failure)
    _append_event(state,{**failure,'component':'paper_fault_audit',
        'status':'audit_incomplete','kind':'implementation','fallback':False})


def decide_and_account(call, state, method, role):
    with capture_http_attempts() as attempts:
        try:
            return call()
        finally:
            state.vlm_http_attempts = getattr(state, 'vlm_http_attempts', 0) + attempts[0]
            key = 'uav_vlm_http_attempts' if role == 'UAV' else 'ugv_vlm_http_attempts'
            setattr(state, key, getattr(state, key, 0) + attempts[0])
            if hasattr(method, 'drain_internal_vlm_calls'):
                count=max(0,int(method.drain_internal_vlm_calls() or 0))
                state.vlm_calls += count
                key='uav_vlm_calls' if role == 'UAV' else 'ugv_vlm_calls'
                setattr(state,key,getattr(state,key)+count)


def http_accounting_result(state):
    return {**{key: getattr(state, key, 0) for key in
               ('vlm_http_attempts', 'uav_vlm_http_attempts', 'ugv_vlm_http_attempts')},
            'vlm_call_accounting': {'budget_unit': 'logical_call',
                'http_attempt_unit': 'generation_post', 'includes_retries': True,
                'tokenizer_preflight_included': False}}


def model_call_accounting_result(state):
    """Retain costs for a crash before an episode result can be constructed."""
    return {**{key: getattr(state, key, 0) for key in
               ('vlm_calls', 'uav_vlm_calls', 'ugv_vlm_calls')},
            **http_accounting_result(state)}


def record_tool_faults(state, observation, trace):
    errors={'error','invalid_source','invalid_frame','infra_error','pipeline_error','transport_error','service_error','partial_error','unavailable'}
    def scan(value, component):
        if isinstance(value,dict):
            # Fixed controllers reject wrong-frame crops safely and expose a
            # counter instead of a status. Preserve that service/frame fault
            # even when the controller continues with another proposal.
            rejected_sources=value.get('source_mismatch',0)
            if isinstance(rejected_sources,(int,float)) and rejected_sources>0:
                _append_event(state,{'step':observation.step,'role':observation.role,
                    'component':component,'status':'invalid_source',
                    'error':str(value.get('source_mismatch_detail',
                        'source_mismatch='+str(rejected_sources)))[:500],
                    'kind':'tool_frame','fallback':True})
            status=value.get('status',value.get('outcome'))
            faulty_ground=value.get('ground_status') in errors
            faulty_sam=value.get('sam_status') in errors
            if faulty_ground:status=value['ground_status']
            elif faulty_sam:status=value['sam_status']
            model_status=next((value.get(key) for key in ('status','protocol_status','parse_status','identity_status')
                if isinstance(value.get(key),str) and value[key] in MODEL_INFRA_STATUSES),None)
            if model_status is not None:
                status=model_status
            if isinstance(status,str) and (status in errors | MODEL_INFRA_STATUSES or status.endswith(('_ground_error','_sam_error'))):
                is_map=component == 'map_metadata' or component.startswith('map_metadata.')
                kind=('model_transport' if status in MODEL_INFRA_STATUSES else 'implementation' if is_map else
                      'tool_frame' if status in ('invalid_source','invalid_frame') else 'tool_service')
                _append_event(state,{'step':observation.step,'role':observation.role,
                    'component':component,'status':status,
                    'error':str(value.get('error') or value.get('ground_error') or value.get('tool_error') or '')[:500],
                    'kind':kind,'fallback':not is_map})
            for key,child in value.items():
                if isinstance(child,(dict,list)):
                    scan(child,str(key) if component == 'method_trace' else component+'.'+str(key))
        elif isinstance(value,list):
            for child in value:scan(child,component)
    scan(trace,'method_trace')
    bundle=getattr(observation,'ugv_evidence_bundle',None)
    if bundle is not None:
        scan(bundle if isinstance(bundle,dict) else vars(bundle),'ugv_evidence')
    metadata=getattr(observation,'map_metadata',None)
    if metadata is not None:
        scan(metadata,'map_metadata')


def finalize_fault_audit(config,state,result):
    result.update(http_accounting_result(state))
    failures=list(getattr(state,'paper_audit_failures',[]) or [])
    complete=getattr(state,'paper_audit_complete',True) is True and not failures
    events=list(getattr(state,'paper_fault_events',[]) or [])
    result['paper_fault_audit']={'complete':complete,'events':events,'failures':failures}
    experiment=config.get('paper_experiment') or {}
    result['paper_experiment']=dict(experiment)
    result['paper_formal_score_eligible']=(complete and not events and not bool(result.get('runtime_error'))
                                         and result.get('completed') is True)
    result['dataset_id']=(config.get('benchmark') or {}).get('dataset_id')
    metadata=result.get('run_meta') or {}
    binding=dict(metadata.get('evaluation_binding') or {})
    binding.update({key:metadata[key] for key in ('plan_sha256','policy_seed','release_sha256') if key in metadata})
    result['evaluation_binding']={key:binding[key] for key in ('plan_sha256','policy_seed','release_sha256') if key in binding}
    return result
