# Knowledge Collapse

Replication code for <a href="https://rdcu.be/d6Qfx" target="_blank">"AI and the Problem of Knowledge Collapse"</a>

## Simulation Analysis

- Generates results for Figures 1-6.

## Empirical Study

- Generates results for Figures 7-9 and Table 3.

- Also includes a small corpus asking for lists of languages mentioned after Definition 3.

- NB: Simple counts from the corpus were generated with grep, e.g.

```
# i = ignore case, r=recursive, E=extended regexp, o=print only matching
grep -iEr 'Sign Language|ASL|BSL|Gesture|Non-verbal|Deaf|Manual|Visual Language' /path-to-folder/languages 

grep -iro 'Aristot' /path-to-folder/corpus | wc -l
```
