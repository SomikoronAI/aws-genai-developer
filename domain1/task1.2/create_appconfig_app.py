# Assignment Part 2

"""
This script automates AWS AppConfig applicatio process.
- Get or create Application, Environment, Configuration Profile
- Create Hosted Configuration Version from a JSON file
- Start a deployment with a chosen deployment strategy
- Optionally wait for completion

Usage:
python create_appconfig_app.py `
--region-name us-east-1 `
--app-name AIAssistantApp `
--config-file ./data/fm_selection_strategy.json `
--wait
"""


import sys
import time
import json
import argparse
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def get_appconfig_client(region_name: str):
    return boto3.client("appconfig", region_name=region_name)


def get_or_create_application(client, app_name: str) -> str:
    paginator = client.get_paginator("list_applications")
    for page in paginator.paginate():
        for item in page.get("Items", []):
            if item.get("Name") == app_name:
                return item["Id"]

    resp = client.create_application(Name=app_name)
    return resp["Id"]


def get_or_create_environment(client, app_id: str, env_name: str) -> str:
    paginator = client.get_paginator("list_environments")
    for page in paginator.paginate(ApplicationId=app_id):
        for env in page.get("Items", []):
            if env.get("Name") == env_name:
                return env["Id"]

    resp = client.create_environment(ApplicationId=app_id, Name=env_name)
    return resp["Id"]


def get_or_create_config_profile(client, app_id: str, profile_name: str) -> str:
    paginator = client.get_paginator("list_configuration_profiles")
    for page in paginator.paginate(ApplicationId=app_id):
        for prof in page.get("Items", []):
            if prof.get("Name") == profile_name:
                return prof["Id"]

    # Hosted configuration profile for freeform JSON
    resp = client.create_configuration_profile(
        ApplicationId=app_id,
        Name=profile_name,
        LocationUri="hosted",
        Type="AWS.Freeform",
    )
    return resp["Id"]


def create_hosted_configuration_version(
    client,
    app_id: str,
    profile_id: str,
    content_bytes: bytes,
    content_type: str = "application/json",
    version_label: Optional[str] = None,
) -> int:
    params = {
        "ApplicationId": app_id,
        "ConfigurationProfileId": profile_id,
        "ContentType": content_type,
        "Content": content_bytes,  # Pass raw bytes in Boto3; no manual base64 needed
    }
    if version_label:
        params["VersionLabel"] = version_label

    resp = client.create_hosted_configuration_version(**params)
    return int(resp["VersionNumber"])


def start_deployment(
    client,
    app_id: str,
    env_id: str,
    profile_id: str,
    version_number: int,
    deployment_strategy_id: str,
    description: Optional[str] = None,
) -> int:
    params = {
        "ApplicationId": app_id,
        "EnvironmentId": env_id,
        "ConfigurationProfileId": profile_id,
        "ConfigurationVersion": str(version_number),
        "DeploymentStrategyId": deployment_strategy_id,
    }
    if description:
        params["Description"] = description

    resp = client.start_deployment(**params)
    return int(resp["DeploymentNumber"])


def wait_for_deployment(
    client,
    app_id: str,
    env_id: str,
    deployment_number: int,
    poll_seconds: int = 5,
    timeout_seconds: int = 900,
) -> str:
    """
    Polls get_deployment until State is one of: COMPLETE, ROLLED_BACK, FAILED.
    Returns the final state.
    """
    start = time.time()
    while True:
        resp = client.get_deployment(
            ApplicationId=app_id,
            EnvironmentId=env_id,
            DeploymentNumber=deployment_number,
        )
        state = resp.get("State")
        pct = resp.get("PercentageComplete")
        print(f"Deployment {deployment_number} state={state}, {pct}% complete")

        if state in ("COMPLETE", "ROLLED_BACK", "FAILED"):
            return state

        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for deployment {deployment_number} to complete"
            )
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Deploy AWS AppConfig hosted config")
    parser.add_argument("--region-name", required=True, help="AWS region name, e.g., us-east-1")
    parser.add_argument("--app-name", required=True, help="AppConfig application name")
    parser.add_argument("--config-file", required=True, 
                        help="Path to JSON config file (will be validated and sent as bytes)",
                        )
    parser.add_argument("--env-name", default="prod", help="Environment name (default: prod)")
    parser.add_argument("--profile-name", default="ModelSelectionStrategy", 
                        help="Configuration profile name (default: ModelSelectionStrategy)",
                        )
    parser.add_argument("--deployment-strategy-id", 
                        default="AppConfig.Linear50PercentEvery30Seconds", 
                        help="Deployment strategy ID (default: AppConfig.Linear50PercentEvery30Seconds)",
                        )
    parser.add_argument("--version-label", default=None, help="Optional version label")
    parser.add_argument("--wait", action="store_true", 
                        help="Wait for deployment to complete (poll until COMPLETE/FAILED/ROLLED_BACK)",
                        )
    
    args = parser.parse_args()

    client = get_appconfig_client(args.region_name)

    try:
        # 1) Get or create the core resources
        app_id = get_or_create_application(client, args.app_name)
        env_id = get_or_create_environment(client, app_id, args.env_name)
        profile_id = get_or_create_config_profile(client, app_id, args.profile_name)

        # 2) Load and validate config file
        with open(args.config_file, "rb") as f:
            content_bytes = f.read()
        # Validate JSON (fail fast with a clear error)
        try:
            json.loads(content_bytes.decode("utf-8"))
        except Exception as e:
            print(f"ERROR: Configuration file is not valid JSON: {args.config_file}\n{e}")
            sys.exit(2)

        # 3) Create a hosted configuration version
        version_number = create_hosted_configuration_version(
            client,
            app_id,
            profile_id,
            content_bytes=content_bytes,
            content_type="application/json",
            version_label=args.version_label,
        )
        print(f"Created hosted configuration version: {version_number}")

        # 4) Start a deployment
        deployment_number = start_deployment(
            client,
            app_id,
            env_id,
            profile_id,
            version_number,
            deployment_strategy_id=args.deployment_strategy_id,
            description=f"Automated deployment of {args.profile_name} {version_number}",
        )
        print(f"Deployment started. DeploymentNumber: {deployment_number}")

        # 5) (Optional) Wait for completion
        if args.wait:
            final_state = wait_for_deployment(client, app_id, env_id, deployment_number)
            print(f"Deployment finished with state: {final_state}")
            if final_state != "COMPLETE":
                sys.exit(1)

        # Summary for pipelines
        print(
            json.dumps(
                {
                    "regionName": args.region_name,
                    "applicationName": args.app_name,
                    "applicationId": app_id,
                    "environmentName": args.env_name,
                    "environmentId": env_id,
                    "profileName": args.profile_name,
                    "profileId": profile_id,
                    "versionNumber": version_number,
                    "deploymentNumber": deployment_number,
                },
                indent=2,
            )
        )
    except (ClientError, BotoCoreError) as e:
        print(f"AWS error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
