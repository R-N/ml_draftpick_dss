def parse_number(s):
    s = s.strip()
    if s == "-":
        return 0
    if s.endswith("%"):
        return 0.01 * float(s[:-1])
    return float(s)

def standardize_name(s):
    return s.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")

def parse_multiple(s):
    return ",".join([rl.strip() for rl in s.split("|")])
