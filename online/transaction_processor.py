import pickle
from collections import deque
from pathlib import Path


class TransactionProcessor:
    """
    Maintains online transaction sequences for every account.

    This module DOES NOT perform inference.
    It only updates behavioural sequences.
    """

    def __init__(

        self,

        sequence_file,

        max_sequence_length=100

    ):

        self.max_sequence_length = max_sequence_length

        with open(sequence_file, "rb") as f:

            self.transaction_sequences = pickle.load(f)

        ####################################################
        # Convert every transaction list into deque
        ####################################################

        for account in self.transaction_sequences:

            history = self.transaction_sequences[account]["transactions"]

            self.transaction_sequences[account]["transactions"] = deque(

                history,

                maxlen=self.max_sequence_length

            )

    ########################################################
    # Add Transaction
    ########################################################

    def add_transaction(

        self,

        sender,

        receiver,

        transaction

    ):

        """
        transaction should contain

        amount
        time_gap
        in_degree
        out_degree
        payment_format
        currency
        """

        ####################################################
        # Sender
        ####################################################

        if sender not in self.transaction_sequences:

            self.transaction_sequences[sender] = {

                "transactions": deque(

                    maxlen=self.max_sequence_length

                ),

                "label": 0

            }

        self.transaction_sequences[sender][

            "transactions"

        ].append(transaction)

        ####################################################
        # Receiver
        ####################################################

        if receiver not in self.transaction_sequences:

            self.transaction_sequences[receiver] = {

                "transactions": deque(

                    maxlen=self.max_sequence_length

                ),

                "label": 0

            }

        self.transaction_sequences[receiver][

            "transactions"

        ].append(transaction)

    ########################################################
    # Get Sequence
    ########################################################

    def get_sequence(

        self,

        account

    ):

        if account not in self.transaction_sequences:

            return []

        return list(

            self.transaction_sequences[account][

                "transactions"

            ]

        )

    ########################################################
    # Save
    ########################################################

    def save(

        self,

        output_file

    ):

        save_dict = {}

        for account in self.transaction_sequences:

            save_dict[account] = {

                "transactions": list(

                    self.transaction_sequences[account][

                        "transactions"

                    ]

                ),

                "label": self.transaction_sequences[account]["label"]

            }

        with open(output_file, "wb") as f:

            pickle.dump(

                save_dict,

                f

            )


############################################################
# Testing
############################################################

if __name__ == "__main__":

    _here = Path(__file__).resolve().parent          # .../online/
    _seq_file = _here.parent / "data" / "processed" / "transaction_sequences.pkl"

    processor = TransactionProcessor(_seq_file)

    sample_transaction = {

        "amount": 1250.0,

        "time_gap": 3600,

        "in_degree": 10,

        "out_degree": 6,

        "payment_format": "Wire",

        "currency": "USD"

    }

    processor.add_transaction(

        sender="ACC001",

        receiver="ACC002",

        transaction=sample_transaction

    )

    print()

    print("=" * 60)

    print("Sender Sequence Length")

    print(

        len(

            processor.get_sequence(

                "ACC001"

            )

        )

    )

    print()

    print("Receiver Sequence Length")

    print(

        len(

            processor.get_sequence(

                "ACC002"

            )

        )

    )

    print()

    print("=" * 60)