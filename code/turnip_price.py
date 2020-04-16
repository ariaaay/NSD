import argparse

def cal(sell_price, entry_fee, num_turnips=4000):
    return (sell_price * num_turnips - entry_fee) / num_turnips

