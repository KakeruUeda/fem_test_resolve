cd ../data/demo/systems

for f in b_NS_*.mtx; do
  echo "Fixing $f"

  N=$(grep -v '^Process \[' "$f" | wc -l)

  {
    echo "%%MatrixMarket matrix array real general"
    echo "% Generated vector for sparse matrix system"
    echo "$N"
    grep -v '^Process \[' "$f"
  } > "$f.fixed"

  mv "$f.fixed" "$f"
done
