# N-gram

See docstrings for the details.

## Instruction

1. Implement add_sentence_tokens
2. Implement replace_unknown
3. Implement _convert_oov
4. Implement perplexity
5. Implement _best_candidate

## Expected Result

```Python
Already exist: ../data/ngram_train.txt
Already exist: ../data/ngram_test.txt
Loading 1-gram model...
Vocabulary size: 23505
10 examples of bigram
[('<s>',), ('liberty',), ('all',), ('star',), ('usa',), ('sets',), ('initial',), ('payout',), ('</s>',), ('<s>',)]
10 examples of FreqDist
[(('<s>',), 60000), (('liberty',), 23), (('all',), 1259), (('star',), 39), (('usa',), 41), (('sets',), 426), (('initial',), 255), (('payout',), 219), (('</s>',), 60000), (('we',), 1392)]
Generating sentences...
<s> the of to in and said a mln for dlrs vs </s> (0.02054)
<s> of to in and said a mln for dlrs vs it </s> (0.01978)
<s> to in and said a mln for dlrs vs it pct of on is from its that at by be cts year will </s> (0.00904)
<s> in and said a mln for dlrs vs it pct on to is from its that at by be cts year will with </s> (0.00890)
<s> and said a mln for dlrs vs it pct on is in from its that at by be cts year will with billion </s> (0.00876)
<s> said a mln for dlrs vs it pct on is from and its that at by be cts year will with billion net </s> (0.00864)
<s> a mln for dlrs vs it pct on is from its said that at by be cts year will with billion net was </s> (0.00853)
<s> mln for dlrs vs it pct on is from its that a at by be cts year will with billion net was us </s> (0.00841)
<s> for dlrs vs it pct on is from its that at mln by be cts year will with billion net was us he </s> (0.00830)
<s> dlrs vs it pct on is from its that at by for be cts year will with billion net was us he has </s> (0.00821)
Model perplexity: 762.939

Loading 2-gram model...
Vocabulary size: 23505
10 examples of bigram
[('<s>', 'liberty'), ('liberty', 'all'), ('all', 'star'), ('star', 'usa'), ('usa', 'sets'), ('sets', 'initial'), ('initial', 'payout'), ('payout', '</s>'), ('</s>', '<s>'), ('<s>', 'we')]
10 examples of FreqDist
[(('<s>', 'liberty'), 9), (('liberty', 'all'), 2), (('all', 'star'), 2), (('star', 'usa'), 1), (('usa', 'sets'), 1), (('sets', 'initial'), 19), (('initial', 'payout'), 13), (('payout', '</s>'), 181), (('</s>', '<s>'), 59999), (('<s>', 'we'), 493)]
Generating sentences...
<s> the company said it has been made a share in 1986 </s> (0.03374)
<s> it said the company also be a share in 1986 87 03 09 pct of its board </s> (0.01975)
<s> shr loss of the company said it has been made a share </s> (0.03131)
<s> he said it has been made a share in the company </s> (0.03295)
<s> in the company said it has been made a share of its board </s> (0.02602)
<s> but the company said it has been made a share in 1986 </s> (0.02921)
<s> a share in the company said it has been made by an agreement to be used for one of its board </s> (0.01582)
<s> us and the company said it has been made a share </s> (0.03029)
<s> this year shr loss of the company said it has been made a share </s> (0.02688)
<s> they said it has been made a share in the company </s> (0.03116)
Model perplexity: 85.795

Loading 3-gram model...
Vocabulary size: 23505
10 examples of bigram
[('<s>', '<s>', 'liberty'), ('<s>', 'liberty', 'all'), ('liberty', 'all', 'star'), ('all', 'star', 'usa'), ('star', 'usa', 'sets'), ('usa', 'sets', 'initial'), ('sets', 'initial', 'payout'), ('initial', 'payout', '</s>'), ('payout', '</s>', '</s>'), ('</s>', '</s>', '<s>')]
10 examples of FreqDist
[(('<s>', '<s>', 'liberty'), 9), (('<s>', 'liberty', 'all'), 2), (('liberty', 'all', 'star'), 2), (('all', 'star', 'usa'), 1), (('star', 'usa', 'sets'), 1), (('usa', 'sets', 'initial'), 1), (('sets', 'initial', 'payout'), 5), (('initial', 'payout', '</s>'), 10), (('payout', '</s>', '</s>'), 181), (('</s>', '</s>', '<s>'), 59999)]
Generating sentences...
<s> <s> the company said it has agreed to sell its shares in a statement </s> (0.03163)
<s> <s> it said the company also announced an offering of up to one billion dlrs in cash and notes </s> (0.01825)
<s> <s> shr loss one ct vs profit two cts net 119 mln dlrs </s> (0.02536)
<s> <s> he said the company also announced an offering of up to one billion dlrs in cash and notes </s> (0.01806)
<s> <s> in a statement that the us agriculture department said it has agreed to sell its shares </s> (0.02298)
<s> <s> but the company said it has agreed to sell its shares in a statement </s> (0.02676)
<s> <s> a spokesman for the first quarter of 1986 and 1985 </s> (0.03440)
<s> <s> us officials said the company also announced an offering of up to one billion dlrs in cash and notes </s> (0.01620)
<s> <s> this is a major trade bill that would be the first quarter of 1986 </s> (0.02190)
<s> <s> they said the company also announced an offering of up to one billion dlrs in cash and notes </s> (0.01751)
Model perplexity: 44.403

Loading 4-gram model...
Vocabulary size: 23505
10 examples of bigram
[('<s>', '<s>', '<s>', 'liberty'), ('<s>', '<s>', 'liberty', 'all'), ('<s>', 'liberty', 'all', 'star'), ('liberty', 'all', 'star', 'usa'), ('all', 'star', 'usa', 'sets'), ('star', 'usa', 'sets', 'initial'), ('usa', 'sets', 'initial', 'payout'), ('sets', 'initial', 'payout', '</s>'), ('initial', 'payout', '</s>', '</s>'), ('payout', '</s>', '</s>', '</s>')]
10 examples of FreqDist
[(('<s>', '<s>', '<s>', 'liberty'), 9), (('<s>', '<s>', 'liberty', 'all'), 2), (('<s>', 'liberty', 'all', 'star'), 2), (('liberty', 'all', 'star', 'usa'), 1), (('all', 'star', 'usa', 'sets'), 1), (('star', 'usa', 'sets', 'initial'), 1), (('usa', 'sets', 'initial', 'payout'), 1), (('sets', 'initial', 'payout', '</s>'), 5), (('initial', 'payout', '</s>', '</s>'), 10), (('payout', '</s>', '</s>', '</s>'), 181)]
Generating sentences...
<s> <s> <s> the company said it will offer a stake in burlington industries inc and gencorp </s> (0.01924)
<s> <s> <s> it said the new agreement will at its option convert to a four pct annual rate in </s> (0.01344)
<s> <s> <s> shr loss five cts vs profit three </s> (0.04892)
<s> <s> <s> he said the company has not yet been determined it will release an announcement this weekend that </s> (0.01414)
<s> <s> <s> in a filing with the securities and exchange commission it has acquired an eight pct coupon to yield 810 </s> (0.01791)
<s> <s> <s> but the company said it will offer a stake in burlington industries inc and gencorp </s> (0.01631)
<s> <s> <s> a spokesman for the new york stock exchange said it will offer up to 100 mln dlrs of debt </s> (0.01517)
<s> <s> <s> us officials said they hope the government also announced an end to agricultural subsidies inclusion of trade in services and investments </s> (0.01011)
<s> <s> <s> this is the first time since 1980 but they predicted either no rise in employment </s> (0.01484)
<s> <s> <s> they said the fed will inject permanent reserves via three day system repurchase agreements economists </s> (0.01574)
Model perplexity: 31.247

Loading 5-gram model...
Vocabulary size: 23505
10 examples of bigram
[('<s>', '<s>', '<s>', '<s>', 'liberty'), ('<s>', '<s>', '<s>', 'liberty', 'all'), ('<s>', '<s>', 'liberty', 'all', 'star'), ('<s>', 'liberty', 'all', 'star', 'usa'), ('liberty', 'all', 'star', 'usa', 'sets'), ('all', 'star', 'usa', 'sets', 'initial'), ('star', 'usa', 'sets', 'initial', 'payout'), ('usa', 'sets', 'initial', 'payout', '</s>'), ('sets', 'initial', 'payout', '</s>', '</s>'), ('initial', 'payout', '</s>', '</s>', '</s>')]
10 examples of FreqDist
[(('<s>', '<s>', '<s>', '<s>', 'liberty'), 9), (('<s>', '<s>', '<s>', 'liberty', 'all'), 2), (('<s>', '<s>', 'liberty', 'all', 'star'), 2), (('<s>', 'liberty', 'all', 'star', 'usa'), 1), (('liberty', 'all', 'star', 'usa', 'sets'), 1), (('all', 'star', 'usa', 'sets', 'initial'), 1), (('star', 'usa', 'sets', 'initial', 'payout'), 1), (('usa', 'sets', 'initial', 'payout', '</s>'), 1), (('sets', 'initial', 'payout', '</s>', '</s>'), 5), (('initial', 'payout', '</s>', '</s>', '</s>'), 10)]
Generating sentences...
<s> <s> <s> <s> the company said it will use proceeds to redeem on june 15 all 125 mln dlrs of subordinated debentures due </s> (0.01193)
<s> <s> <s> <s> it said the new agreement will at its option convert to a four year term loan in september 1988 </s> </s> (0.01042)
<s> <s> <s> <s> shr loss five cts vs profit eight </s> (0.04413)
<s> <s> <s> <s> he said the company has not yet filed a plan of reorganization bell is also free to continue talks with </s> (0.01075)
<s> <s> <s> <s> in a filing with the securities and exchange commission it has acquired 150000 shares of modulaire industries or 50 pct </s> (0.01528)
<s> <s> <s> <s> but the company said it will use proceeds to redeem on june 15 all 125 mln dlrs of subordinated debentures </s> (0.01087)
<s> <s> <s> <s> a spokesman for the new york fed said </s> (0.03340)
<s> <s> <s> <s> us officials said they were giving japan until april to show that an economic stimulus package was in the offing </s> (0.01077)
<s> <s> <s> <s> this is the first time a japanese bank has bought </s> (0.02536)
<s> <s> <s> <s> they said the fed is reluctant to lower short term rates for fear this would spur expectations of a weaker </s> (0.01020)
Model perplexity: 45.623
```

If you don't implement the code but just try to mimic answers above. You will get zero for this task.

## Q&A

Check a notification in KLMS.
