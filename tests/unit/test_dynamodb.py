"""Tests for DynamoDB single-table client."""

from unittest.mock import MagicMock, patch

import pytest

from travel_planner.data.dynamodb import DynamoDBClient


@pytest.fixture
def mock_boto3():
    with patch("travel_planner.data.dynamodb.boto3") as mock:
        mock_table = MagicMock()
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock.resource.return_value = mock_resource
        yield mock, mock_table


def test_client_init_local(mock_boto3):
    mock, mock_table = mock_boto3
    client = DynamoDBClient(
        table_name="test-table",
        endpoint_url="http://localhost:8000",
        region="ap-northeast-1",
    )
    assert client.table_name == "test-table"
    mock.resource.assert_called_once()


def test_client_init_aws(mock_boto3):
    mock, mock_table = mock_boto3
    client = DynamoDBClient(
        table_name="test-table",
        region="ap-northeast-1",
    )
    assert client.table_name == "test-table"


def test_put_item(mock_boto3):
    mock, mock_table = mock_boto3
    client = DynamoDBClient(table_name="test-table", region="ap-northeast-1")
    client.put_item({"PK": "USER#123", "SK": "PROFILE", "Data": {"name": "Test"}})
    mock_table.put_item.assert_called_once()


def test_get_item(mock_boto3):
    mock, mock_table = mock_boto3
    mock_table.get_item.return_value = {
        "Item": {"PK": "USER#123", "SK": "PROFILE", "Data": {"name": "Test"}}
    }
    client = DynamoDBClient(table_name="test-table", region="ap-northeast-1")
    item = client.get_item("USER#123", "PROFILE")
    assert item["PK"] == "USER#123"


def test_get_item_not_found(mock_boto3):
    mock, mock_table = mock_boto3
    mock_table.get_item.return_value = {}
    client = DynamoDBClient(table_name="test-table", region="ap-northeast-1")
    item = client.get_item("USER#999", "PROFILE")
    assert item is None


def test_query(mock_boto3):
    mock, mock_table = mock_boto3
    mock_table.query.return_value = {
        "Items": [
            {"PK": "USER#123#CONVERSATION", "SK": "CONV#1"},
            {"PK": "USER#123#CONVERSATION", "SK": "CONV#2"},
        ]
    }
    client = DynamoDBClient(table_name="test-table", region="ap-northeast-1")
    items = client.query(pk="USER#123#CONVERSATION")
    assert len(items) == 2


def test_query_with_sk_prefix(mock_boto3):
    mock, mock_table = mock_boto3
    mock_table.query.return_value = {"Items": []}
    client = DynamoDBClient(table_name="test-table", region="ap-northeast-1")
    items = client.query(pk="USER#123", sk_prefix="SESSION#")
    assert items == []
    call_kwargs = mock_table.query.call_args[1]
    assert "KeyConditionExpression" in call_kwargs


def test_delete_item(mock_boto3):
    mock, mock_table = mock_boto3
    client = DynamoDBClient(table_name="test-table", region="ap-northeast-1")
    client.delete_item("USER#123", "PROFILE")
    mock_table.delete_item.assert_called_once()
