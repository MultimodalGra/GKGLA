def get_model(model_name, args):
    if model_name.lower() != "gkgla":
        raise ValueError(f"Unsupported model: {model_name}; expected gkgla")
    from models.gkgla import GKGLALearner

    return GKGLALearner(args)
