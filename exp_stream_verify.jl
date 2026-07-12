include("src/cc_cpu.jl")
include("src/cc_gpu_wheel.jl")

# 既知 CC15 先頭が stream 拡張wheel(31,47追加) の生存残基に残るか (CPU即時証明)
cc_len_mod(x, p, k) = (n=0; y=x; while n<k && y%p!=0; y=(2y+1)%p; n+=1; end; n)

head = Int128(90616211958465842219)   # 既知最小 CC15 (9.06e19)
k = 15
wl = _cc_wl_stream(k)
println("CC15 head=$head")
println("stream wl=$wl")
survives = true
for p in wl
    r = Int(head % Int128(p))
    L = cc_len_mod(r, p, k)
    bad = L < k
    bad && (global survives = false)
    println("  p=$p  head%p=$r  chainlen_mod=$L  ", bad ? "<-- BAD (除外!)" : "ok")
end
println(survives ? "=> head は wheel を通過する (取りこぼさない)" : "=> head が除外される (BUG!)")

# 参考: 大レンジ(≫wheel)での実測ベース速度解析
println("\n--- 速度解析 (先の 5e16 レンジ実測から) ---")
ext_thr = 3.03e10/56.8;  ext_den = 6.08e-7
str_thr = 2.07e11/528.0; str_den = 3.36e-7
ext_unit = ext_den/ext_thr
str_unit = str_den/str_thr
println("  ext(29)  : $(round(ext_thr/1e8,digits=2))e8 threads/s, density=$ext_den")
println("  stream   : $(round(str_thr/1e8,digits=2))e8 threads/s, density=$str_den")
println("  大レンジ speedup stream/ext = $(round(ext_unit/str_unit, digits=3))x")
println("  (注: レンジ幅 ≫ wheel=6.15e17 のとき有効)")
