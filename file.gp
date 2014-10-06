set term png
set output "grafic.png"
set title "Tema4 ASC"
set xlabel "Dimensiune"
set ylabel "Timp(s)"
set grid
set style line 22 linetype 3 linewidth 4
plot "grafic.txt" using 4:1 title "GPU NoShared" with linespoints, "grafic.txt" using 4:2 title "GPU Shared" with linespoints, "grafic.txt" using 4:3 title "CPU" with linespoints
