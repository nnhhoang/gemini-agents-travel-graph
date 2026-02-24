"""
DynamoDB single-table client.

Supports both DynamoDB Local (development) and AWS DynamoDB (production).
Set DYNAMODB_ENDPOINT env var for local, omit for AWS.
"""

from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class DynamoDBClient:
    """Client for DynamoDB single-table operations."""

    def __init__(
        self,
        table_name: str,
        region: str = "ap-northeast-1",
        endpoint_url: str | None = None,
    ):
        self.table_name = table_name
        kwargs: dict[str, Any] = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
            logger.info(f"Using DynamoDB Local at {endpoint_url}")

        resource = boto3.resource("dynamodb", **kwargs)
        self.table = resource.Table(table_name)

    def put_item(self, item: dict[str, Any]) -> None:
        """Put an item into the table."""
        self.table.put_item(Item=item)

    def get_item(self, pk: str, sk: str) -> dict[str, Any] | None:
        """Get a single item by PK and SK."""
        response = self.table.get_item(Key={"PK": pk, "SK": sk})
        return response.get("Item")

    def query(
        self,
        pk: str,
        sk_prefix: str | None = None,
        sk_between: tuple[str, str] | None = None,
        index_name: str | None = None,
        limit: int | None = None,
        scan_forward: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query items by partition key with optional sort key conditions.

        Args:
            pk: Partition key value
            sk_prefix: Sort key prefix (begins_with)
            sk_between: Sort key range (between t1 and t2)
            index_name: GSI name (e.g., "GSI1")
            limit: Max items to return
            scan_forward: True for ascending, False for descending
        """
        pk_attr = "GSI1PK" if index_name else "PK"
        sk_attr = "GSI1SK" if index_name else "SK"

        key_condition = Key(pk_attr).eq(pk)

        if sk_prefix:
            key_condition = key_condition & Key(sk_attr).begins_with(sk_prefix)
        elif sk_between:
            key_condition = key_condition & Key(sk_attr).between(*sk_between)

        kwargs: dict[str, Any] = {
            "KeyConditionExpression": key_condition,
            "ScanIndexForward": scan_forward,
        }
        if index_name:
            kwargs["IndexName"] = index_name
        if limit:
            kwargs["Limit"] = limit

        response = self.table.query(**kwargs)
        return response.get("Items", [])

    def delete_item(self, pk: str, sk: str) -> None:
        """Delete an item by PK and SK."""
        self.table.delete_item(Key={"PK": pk, "SK": sk})

    def update_item(
        self,
        pk: str,
        sk: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update specific attributes of an item."""
        expr_parts = []
        expr_values = {}
        expr_names = {}

        for i, (key, value) in enumerate(updates.items()):
            placeholder = f":val{i}"
            name_placeholder = f"#attr{i}"
            expr_parts.append(f"{name_placeholder} = {placeholder}")
            expr_values[placeholder] = value
            expr_names[name_placeholder] = key

        response = self.table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression="SET " + ", ".join(expr_parts),
            ExpressionAttributeValues=expr_values,
            ExpressionAttributeNames=expr_names,
            ReturnValues="ALL_NEW",
        )
        return response.get("Attributes", {})

    def batch_write(self, items: list[dict[str, Any]]) -> None:
        """Batch write items (max 25 per call)."""
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)

    def query_gsi1(
        self,
        gsi1pk: str,
        sk_prefix: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query GSI1 index."""
        return self.query(
            pk=gsi1pk,
            sk_prefix=sk_prefix,
            index_name="GSI1",
            limit=limit,
        )

    def create_table_if_not_exists(self) -> None:
        """Create the table (for DynamoDB Local development)."""
        try:
            self.table.load()
            logger.info(f"Table {self.table_name} already exists")
        except Exception:
            resource = self.table.meta.client
            resource.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "SK", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                    {"AttributeName": "GSI1PK", "AttributeType": "S"},
                    {"AttributeName": "GSI1SK", "AttributeType": "S"},
                ],
                GlobalSecondaryIndexes=[
                    {
                        "IndexName": "GSI1",
                        "KeySchema": [
                            {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                            {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    }
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            logger.info(f"Created table {self.table_name}")
