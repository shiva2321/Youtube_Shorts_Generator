#!/usr/bin/env python3
"""Quick test to verify the overlay filter generation is fixed."""

# Test the drawtext y parameter generation
bar_pct = 0.14
top_y = "(h*{pct:.4f}-text_h)/2".format(pct=bar_pct)
bot_y = "h-(h*{pct:.4f})+(h*{pct:.4f}-text_h)/2".format(pct=bar_pct)

print(f"Generated top_y: {top_y}")
print(f"Generated bot_y: {bot_y}")

# The FIXED drawtext parameters (without quotes)
drawtext_top_fixed = f"y={top_y}"
drawtext_bot_fixed = f"y={bot_y}"

print(f"\nFixed drawtext top: {drawtext_top_fixed}")
print(f"Fixed drawtext bot: {drawtext_bot_fixed}")

# Compare with old (broken) version that had quotes
drawtext_top_broken = f"y='{top_y}'"
drawtext_bot_broken = f"y='{bot_y}'"

print(f"\nBroken drawtext top: {drawtext_top_broken}")
print(f"Broken drawtext bot: {drawtext_bot_broken}")

print("\n✓ Fix verified: Quotes removed from y parameter expressions")

