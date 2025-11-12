```mermaid
flowchart TD
  %% Start / decision
  A((Start)) --> B{size n == 1?}

  %% Base case nodes
  B -->|True| C[" C[0][0] = A[0][0] × B[0][0]"]
  C --> D["Return C"]
  D --> Z((End))

  %% Recursive entry
  B  -->|False| E["1. Seperate 4 sub-matrix (n/2 × n/2)<br/>A11, A12, A21, A22 / B11, B12, B21, B22"]

  %% 7 recursive multiplications (P1..P7)
  E --> P1["P1 = Strassen(A11, B12 - B22)"]
  P1 --> P2["P2 = Strassen(A11 + A12, B22)"]
  P2 --> P3["P3 = Strassen(A21 + A22, B11)"]
  P3 --> P4["P4 = Strassen(A22, B21 - B11)"]
  P4 --> P5["P5 = Strassen(A11 + A22, B11 + B22)"]
  P5 --> P6["P6 = Strassen(A12 - A22, B21 + B22)"]
  P6 --> P7["P7 = Strassen(A11 - A21, B11 + B12)"]

  %% Combine results into C11..C22
  P7 --> C11["C11 = P5 + P4 - P2 + P6"]
  C11 --> C12["C12 = P1 + P2"]
  C12 --> C21["C21 = P3 + P4"]
  C21 --> C22["C22 = P5 + P1 - P3 - P7"]

  %% Final combine and return
  C22 --> G["4. combine C11, C12, C21, C22 -> C"]
  G --> H["Return C"]
  H --> Z
