import os
import random
import sys
import time
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
    for _ in range(random.randint(1, len(user_ids) // 4)):
        u0, u1 = random.sample(user_ids, 2)
        friendlist[u0] = friendship(u0, u1, random.uniform(.5, 1))
        print(f"{u0} and {u1} are friends with weight .{int(100*friendlist[u0].weight)}")
    friendly_users = friendlist.keys()

    print("Generating samples")
    starttime = int(time.time())
    for _ in range(samples):
        member = random.choice(user_ids)
        others = list(user_ids)
        others.remove(member)
        present = random.sample(others, random.randint(1, len(others)))
        if member in friendly_users and friendlist[member].u1 not in present:
            if random.random() < friendlist[member].weight:
                present.append(friendlist[member].u1)

        df = df.append(pandas.Series({"member": str(member), "present": present}, name=starttime))
        starttime += random.randint(1, 1000)

    print("Data successfully generated!")
    print("------")
    print(df)

    print(f"Saving data to {savepath}.")
    df.to_csv(savepath)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_data.py <samples> <num_members> <filename>")
        exit(-1)
    samples = int(sys.argv[1])
    num_members = int(sys.argv[2])
    savepath = os.getcwd() + '\\' + str(sys.argv[3])
    generate_samples(samples, num_members, savepath)
