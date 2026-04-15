import matplotlib.pyplot as plt


def main():
    num = None
    title = ""
    groups = []
    with open("time_data.txt", "r") as rf:
        for line in rf.readlines():
            word = line.split()
            if not word:
                continue
            if word[0].startswith("_"):
                if num is not None:
                    groups.append((title, num))
                title = word[0].strip()
                num = []
            elif word[0].startswith("d"):
                try:
                    num.append(float(word[1]))
                except (ValueError, IndexError):
                    continue

    if num is not None:
        groups.append((title, num))

    if not groups:
        return

    plt.figure()
    for curve_title, curve_num in groups:
        if not curve_num:
            continue
        x = list(range(1, len(curve_num) + 1))
        legend_label = curve_title.lstrip("_") or curve_title
        plt.plot(x, curve_num, marker="o", label=legend_label)
    plt.title("deltatime_detection comparison")
    plt.xlabel("num")
    plt.ylabel("deltatime_detection (s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=14)
    plt.tight_layout(rect=[0.0, 0.0, 1, 1])

    plt.show()


if __name__ == "__main__":
    main()
