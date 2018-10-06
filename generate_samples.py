import os
import random
import time
import argparse
from collections import namedtuple

import pandas

friendship = namedtuple('friendship', 'u0 u1 weight')


def generate_samples(samples, num_members, savepath):
    df = pandas.DataFrame({"member": [], "present": []})
    df.index.name = "timestamp"
    user_ids = ["user{}".format(n) for n in range(num_members)]
    print(f"Created {num_members} members.")

    # Create some random friendships
    print("Generating friends...")
    friendlist = {}
    for _ in range(int(random.triangular(1, len(user_ids), 2))):
        # BUG: Sometimes u0 and u1 are duplicates that overwrite previous values
        u0, u1 = random.sample(user_ids, 2)
        friendlist[u0] = friendship(u0, u1, random.uniform(.7, 1))
        print(f"{u0} and {u1} are friends with weight .{int(100*friendlist[u0].weight)}")
    friendly_users = friendlist.keys()

    print("Generating samples")
    current_time = int(time.time())
    for _ in range(samples):
        member = random.choice(user_ids)
        others = list(user_ids)
        others.remove(member)
        present = random.sample(others, int(random.triangular(1, len(others), 2)))
        if member in friendly_users and friendlist[member].u1 not in present:
            # Friend isn't in the server; check on adding them
            if random.random() < friendlist[member].weight:
                present.append(friendlist[member].u1)

        df = df.append(pandas.Series({"member": str(member), "present": present}, name=current_time))
        # increment time for next "event"
        current_time += random.randint(1, 1000)

    print("Data successfully generated!")
    print("------")
    print(df)

    print(f"Saving data to {savepath}")
    df.to_csv(savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help="The number of samples to generate", type=int)
    parser.add_argument("num_members", help="The number of distinct members in the simulated server", type=int)
    parser.add_argument("filename", help="The name of the output file. Will output in the current working directory.")
    args = parser.parse_args()

    savepath = os.getcwd() + '\\' + args.filename
    generate_samples(args.samples, args.num_members, savepath)
