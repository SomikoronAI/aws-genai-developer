# Assignment Part 3


$account_id="339713184295"
$region1_name="us-east-1"
$region2_name="us-west-2"
$stack_name="AIAssistantStack"
$deployment_template="crossregion_deployment_template.yaml"
$lambda_function_name="AIAssistantPrimaryModel"
$state_machine_name="AIAssistantCircuitBreaker"


# Deploy to primary region
aws cloudformation deploy `
--template-file $deployment_template `
--stack-name  $stack_name `
--parameter-overrides `
ModelAbstractionLambda=arn:aws:lambda:${region1_name}:${account_id}:function:${lambda_function_name} `
StateMachineArn=arn:aws:states:${region1_name}:${account_id}:stateMachine:${state_machine_name} `
ApiGatewayRoleArn=arn:aws:iam::${account_id}:role/aws-apigateway-execution-role `
Environment=prod `
--region $region1_name `
--capabilities CAPABILITY_IAM


# Deploy to secondary region
aws cloudformation deploy `
--template-file $deployment_template `
--stack-name $stack_name `
--parameter-overrides `
ModelAbstractionLambda=arn:aws:lambda:${region2_name}:${account_id}:function:${lambda_function_name} `
StateMachineArn=arn:aws:states:${region2_name}:${account_id}:stateMachine:${state_machine_name} `
ApiGatewayRoleArn=arn:aws:iam::${account_id}:role/aws-apigateway-execution-role `
Environment=prod `
--region $region2_name `
--capabilities CAPABILITY_IAM
