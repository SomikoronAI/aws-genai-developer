# Assignment Part 2

def select_model(config, use_case=None):
    """Select appropriate model based on configuration and use case."""
    # # Check if there's a use case specific model
    # use_case_models = config.get('use_case_models', {})
    # if use_case in use_case_models:
    #     return use_case_models[use_case]
    
    # # Default to primary model
    # return config.get('primary_model')

    use_cases = config.get("use_cases", {})

    # Always require general to exist
    general = use_cases.get("general", {})
    if not general or "primary_model" not in general:
        raise ValueError("General primary model is not configured")

    # No use case selected, return general primary model
    if not use_case:
        return general["primary_model"]
    
    # Pick use-case-specific config 
    selected = use_cases.get(use_case)
    if not selected or "primary_model" not in selected:
        print(f"Primary model not configured for use case: {use_case}")
        print("Returning the default model")
        return use_cases.get("default").get("primary_model")
        # raise ValueError(f"Primary model not configured for use case: {use_case}")

    return selected["primary_model"]

