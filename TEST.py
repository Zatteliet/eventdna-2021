from sklearn.metrics import classification_report

# g = ["B", "I", "O", "O"]
# p = ["B", "I", "I", "O"]

g = ["O", "O", "O", "O"]
p = ["O", "O", "O", "O"]

# g = ["O", "O", "O", "O"]
# p = ["B", "I", "I", "O"]

# g = ["B", "I", "I", "O"]
# p = ["O", "O", "O", "O"]


# report = classification_report(g, p)
report = classification_report(g, p, labels=["I", "B", "O"], zero_division=0)
# report = classification_report(g, p, labels=["I", "B", "O"])

print(report)

cases = [
    (["F"], ["F"]),
    (["F"], ["NF"]),
    (["NF"], ["F"]),
    (["NF"], ["NF"]),
]
for gold, pred in cases:
    zd = 1
    print(gold, pred)
    print(classification_report(gold, pred, labels=["F", "NF"], zero_division=zd))
