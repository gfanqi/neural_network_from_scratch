def laji(the_list, indent=False, level=0, is_show=False):
    items = []
    for each_item in the_list:
        if isinstance(each_item, list):
            a = laji(each_item, indent=indent, level=level + 1, is_show=is_show)
            items.extend(a)
        else:
            if is_show:
                if indent:
                    for tab_stop in range(level):
                        print("\t", end="")
                print(each_item)
            items.append(each_item)
    return items




if __name__ == "__main__":
    a=list(range(10))
    c = laji(a)
    print(c)
