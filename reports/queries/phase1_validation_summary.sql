SELECT
    '패턴' AS target,
    0.494423 AS aucpr,
    0.810170 AS auroc,
    0.049752 AS f1,
    2592000 AS support,
    'oracle/sanity' AS status
UNION ALL
SELECT
    '5단계 평균',
    0.639080,
    0.866254,
    0.603361,
    66124,
    'oracle/sanity'
UNION ALL
SELECT
    '행동 평균',
    0.398993,
    0.540588,
    0.379214,
    74322,
    'oracle/sanity';
