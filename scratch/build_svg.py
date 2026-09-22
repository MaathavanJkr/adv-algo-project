import csv
import math

def build_svg():
    csv_file = 'results/summary.csv'
    out_file = 'plots/adversarial_max_chain.svg'
    
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['dataset_name'] == 'adversarial':
                data.append({
                    'n': int(row['n']),
                    'max_chain': float(row['max_chain_length_mean']),
                    'strategy': row['hash_strategy']
                })
                
    if not data:
        print("No adversarial data found")
        return
        
    n_values = sorted(list(set(d['n'] for d in data)))
    min_n = min(n_values)
    max_n = max(n_values)
    
    max_y = max(d['max_chain'] for d in data) * 1.1
    if max_y == 0:
        max_y = 1
        
    # coordinates for mapping
    # x from 80 to 520
    # y from 340 to 60
    def map_x(n):
        if max_n == min_n:
            return 300
        return 80 + 440 * (n - min_n) / (max_n - min_n)
        
    def map_y(val):
        return 340 - 280 * val / max_y
        
    svg = []
    svg.append('<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append('<text x="300.0" y="30" font-family="Arial" font-size="16" text-anchor="middle">Adversarial - Max Chain Length</text>')
    svg.append('<line x1="80" y1="340" x2="520" y2="340" stroke="black" stroke-width="2"/>')
    svg.append('<line x1="80" y1="60" x2="80" y2="340" stroke="black" stroke-width="2"/>')
    svg.append('<text x="300.0" y="380" font-family="Arial" font-size="12" text-anchor="middle">n (number of keys)</text>')
    svg.append('<text x="20" y="200.0" font-family="Arial" font-size="12" text-anchor="middle" transform="rotate(-90, 20, 200.0)">Max Chain Length</text>')
    
    colors = {'fixed': '#1f77b4', 'universal': '#ff7f0e'}
    
    for strategy in ['fixed', 'universal']:
        strat_data = [d for d in data if d['strategy'] == strategy]
        
        # Calculate averages for the line
        averages = []
        for n in n_values:
            n_data = [d['max_chain'] for d in strat_data if d['n'] == n]
            if n_data:
                averages.append((n, sum(n_data) / len(n_data)))
                
        # Draw circles for ALL data points
        for d in strat_data:
            cx = map_x(d['n'])
            cy = map_y(d['max_chain'])
            # Add some jitter to x to separate overlapping points
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{colors[strategy]}" opacity="0.6"/>')
            
        # Draw polyline for average
        if averages:
            points = " ".join([f"{map_x(n)},{map_y(avg)}" for n, avg in averages])
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[strategy]}" stroke-width="2"/>')
            
    # Legend
    svg.append('<rect x="530" y="60" width="10" height="10" fill="#1f77b4"/>')
    svg.append('<text x="545" y="70" font-family="Arial" font-size="12">fixed</text>')
    svg.append('<rect x="530" y="80" width="10" height="10" fill="#ff7f0e"/>')
    svg.append('<text x="545" y="90" font-family="Arial" font-size="12">universal</text>')
    svg.append('</svg>')
    
    with open(out_file, 'w') as f:
        f.write("\n".join(svg) + "\n")
        
if __name__ == '__main__':
    build_svg()
