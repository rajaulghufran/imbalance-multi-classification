SUB_PATTERNS_1 = [
    # remove html tags
    ("<[^>]+>", ""),

    # remove emails
    ("\S*@\S*\s?", ""),

    # remove urls
    ("(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*", "")
]

SUB_PATTERNS_2 = [
    # remove punctuations
    ("[^\s\w]", " "),

    # remove digits
    ("\d", ""),

    # remove single character
    ("\\b\w\\b", ""),

    # normalize multiple whitespace
    ("\s{2,}", " ")
]
