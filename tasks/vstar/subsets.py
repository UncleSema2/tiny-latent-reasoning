"""Select V* categories after validating the complete dataset split."""


def direct_attributes(dataset):
    return dataset.filter("direct_attributes".__eq__, input_columns="category")


def relative_position(dataset):
    return dataset.filter("relative_position".__eq__, input_columns="category")
