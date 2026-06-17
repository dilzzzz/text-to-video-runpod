# Empty — model loads at runtime (not at build time)
# This keeps Docker image small and build fast
print("[BUILD] Skipping model pre-download. Model will load on first request.")
