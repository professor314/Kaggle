"""Cloud resource scaling for Kaggle ML Toolkit (Phase 3 - Planned).

This module will provide AWS CLI integration for:
- Spinning up EC2 instances for GPU training
- S3 storage for datasets and model artifacts
- SageMaker integration for managed training jobs
- Cost estimation and budget management

Prerequisites (when implemented):
- AWS CLI configured with valid credentials
- boto3 Python SDK
- Appropriate IAM permissions

See ROADMAP.md for planned timeline and contribution opportunities.
"""


class CloudScaler:
    """Cloud resource management (Phase 3 - Not yet implemented).
    
    Will support scaling model training to AWS cloud resources,
    managing compute instances, and syncing data/artifacts.
    """

    def __init__(self):
        raise NotImplementedError(
            "Cloud scaling support is planned for Phase 3. "
            "See ROADMAP.md for details and contribution opportunities. "
            "For now, all training runs locally."
        )
