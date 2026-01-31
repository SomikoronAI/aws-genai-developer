# Deploy to primary region
aws cloudformation deploy \
    --template-file template.yaml \
    --stack-name ai-assistant-stack \
    --parameter-overrides Environment=prod \
    --region us-east-1 \
    --capabilities CAPABILITY_IAM

# Deploy to secondary region
aws cloudformation deploy \
    --template-file template.yaml \
    --stack-name ai-assistant-stack \
    --parameter-overrides Environment=prod \
    --region us-west-2 \
    --capabilities CAPABILITY_IAM
