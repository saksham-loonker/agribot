"""Restore exactly the trainable parameter subset, preserving all frozen state."""

def restore_trainable_exact(model, state):
    names={name for name,value in model.named_parameters() if value.requires_grad}
    if set(state)!=names:
        raise ValueError('Checkpoint trainable parameter contract changed')
    merged=model.state_dict()
    for name,value in state.items():
        if value.shape!=merged[name].shape or value.dtype!=merged[name].dtype:
            raise ValueError(f'Checkpoint shape/dtype changed: {name}')
        merged[name]=value
    # A complete strict load avoids PyTorch's backward-compatibility suppression
    # of missing BatchNorm num_batches_tracked keys in partial loads.
    model.load_state_dict(merged,strict=True)
