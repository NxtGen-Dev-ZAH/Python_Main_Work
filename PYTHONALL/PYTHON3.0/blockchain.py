import hashlib
import time
import json


class Block:
    def __init__(
        self,
        index,
        previous_hash,
        timestamp,
        file_hash,
        data,
        public_key=None,
        signature=None,
        hash=None,
    ):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.file_hash = file_hash
        self.data = data
        self.public_key = public_key
        self.signature = signature
        self.hash = hash


def calculate_hash(
    index, previous_hash, timestamp, file_hash, data, public_key=None, signature=None
):
    value = (
        str(index)
        + str(previous_hash)
        + str(timestamp)
        + str(file_hash)
        + str(data)
        + str(public_key)
        + str(signature)
    )
    return hashlib.sha256(value.encode()).hexdigest()


def sign_data(private_key, data):
    signature = hashlib.sha256((private_key + data).encode()).hexdigest()
    return signature


def create_genesis_block():
    return Block(
        0,
        "0",
        time.time(),
        "Genesis_File_Hash",
        "Genesis Block Data",
        "Public_Key_1",
        "Signature_1",
        calculate_hash(
            0,
            "0",
            time.time(),
            "Genesis_File_Hash",
            "Genesis Block Data",
            "Public_Key_1",
            "Signature_1",
        ),
    )


def create_new_block(previous_block, private_key, file_content, additional_data):
    index = previous_block.index + 1
    timestamp = time.time()
    file_hash = hashlib.sha256(file_content.encode()).hexdigest()
    data = f"Additional Data: {additional_data}"
    public_key = "SIMPLE_ZAHEER+MOIZ+HAMMAD_PUBLIC_KEY" + str(index)
    signature = sign_data(private_key, data)
    hash = calculate_hash(
        index, previous_block.hash, timestamp, file_hash, data, public_key, signature
    )
    return Block(
        index,
        previous_block.hash,
        timestamp,
        file_hash,
        data,
        public_key,
        signature,
        hash,
    )


blockchain = [create_genesis_block()]
previous_block = blockchain[0]
file_contents = ["ZAHEER AHMED FILE", "MOIZ ABDULLAH ABBASI FILE", "HAMMAD YOUNUS FILE"]
additional_data = [
    ("AGE OF ZAHEER IS 20", "PYTHON ENTHUSIAST"),
    "AGE OF MOIZ IS 20",
    "AGE OF HAMMAD IS 20",
]
private_key = "12345$$55"

for i, (content, data) in enumerate(zip(file_contents, additional_data)):
    new_block = create_new_block(previous_block, private_key, content, data)
    blockchain.append(new_block)
    previous_block = new_block
    print(
        f"Block #{i + 1} has been added to the blockchain with file hash: {new_block.file_hash}"
    )

for block in blockchain:
    print("\nBlock #", block.index)
    print("Timestamp: ", block.timestamp)
    print("File Hash: ", block.file_hash)
    print("Data: ", block.data)
    print("Public Key: ", block.public_key)
    print("Signature: ", block.signature)
    print("Previous Hash: ", block.previous_hash)
    print("Hash: ", block.hash)
