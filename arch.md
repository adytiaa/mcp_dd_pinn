graph TD
    %% --- Overall Problem & Inputs ---
    A[Problem Definition <br/> (PDE, Global Domain Ω, BCs/ICs)] --> B;
    B{Domain Decomposer};
    D[Coarse Model <br/> Solver/Provider <br/> (e.g., simplified PDE, low-fidelity solver, pre-trained NN)];
    D --> E[u_coarse(x,t) <br/> (Global Coarse Solution)];

    %% --- Domain Decomposition ---
    B --> Sub1[Subdomain Ω₁];
    B --> Sub2[Subdomain Ω₂];
    B --> SubN[Subdomain ΩN <br/> ...];

    %% --- Processing for Subdomain 1 ---
    subgraph Subdomain 1 Processing
        direction LR
        E1[u_coarse |<sub>Ω₁</sub> <br/> (Coarse Sol. in Ω₁)]
        NN1[Correction NN₁ <br/> δu<sub>NN₁</sub>(x,t; θ₁)];
        Combine1["u<sub>total₁</sub> = u<sub>coarse</sub>|<sub>Ω₁</sub> + δu<sub>NN₁</sub>"];
        LossPDE1[PDE Residual Loss <br/> L<sub>PDE₁</sub> for u<sub>total₁</sub>];
        LossBC1[Boundary Cond. Loss <br/> L<sub>BC₁</sub> for u<sub>total₁</sub>];
        %% LossIC1[Initial Cond. Loss <br/> L<sub>IC₁</sub> for u<sub>total₁</sub>]; %% If applicable
    end
    Sub1 --> NN1;
    E --> E1;
    E1 --> Combine1;
    NN1 --> Combine1;
    Combine1 --> LossPDE1;
    Combine1 --> LossBC1;
    %% Combine1 --> LossIC1;

    %% --- Processing for Subdomain 2 ---
    subgraph Subdomain 2 Processing
        direction LR
        E2[u_coarse |<sub>Ω₂</sub> <br/> (Coarse Sol. in Ω₂)]
        NN2[Correction NN₂ <br/> δu<sub>NN₂</sub>(x,t; θ₂)];
        Combine2["u<sub>total₂</sub> = u<sub>coarse</sub>|<sub>Ω₂</sub> + δu<sub>NN₂</sub>"];
        LossPDE2[PDE Residual Loss <br/> L<sub>PDE₂</sub> for u<sub>total₂</sub>];
        LossBC2[Boundary Cond. Loss <br/> L<sub>BC₂</sub> for u<sub>total₂</sub>];
    end
    Sub2 --> NN2;
    E --> E2;
    E2 --> Combine2;
    NN2 --> Combine2;
    Combine2 --> LossPDE2;
    Combine2 --> LossBC2;

    %% --- Processing for Subdomain N ---
    subgraph Subdomain N Processing
        direction LR
        EN[u_coarse |<sub>ΩN</sub> <br/> (Coarse Sol. in ΩN)]
        NNN[Correction NN<sub>N</sub> <br/> δu<sub>NN<sub>N</sub></sub>(x,t; θ<sub>N</sub>)];
        CombineN["u<sub>total<sub>N</sub></sub> = u<sub>coarse</sub>|<sub>ΩN</sub> + δu<sub>NN<sub>N</sub></sub>"];
        LossPDEN[PDE Residual Loss <br/> L<sub>PDE<sub>N</sub></sub> for u<sub>total<sub>N</sub></sub>];
        LossBCN[Boundary Cond. Loss <br/> L<sub>BC<sub>N</sub></sub> for u<sub>total<sub>N</sub></sub>];
    end
    SubN --> NNN;
    E --> EN;
    EN --> CombineN;
    NNN --> CombineN;
    CombineN --> LossPDEN;
    CombineN --> LossBCN;

    %% --- Interface Losses ---
    Combine1 --> LossInterface12{Interface Loss <br/> L<sub>int₁₂</sub> <br/> (u<sub>total₁</sub>, u<sub>total₂</sub> on Γ₁₂)};
    Combine2 --> LossInterface12;

    %% Example for another interface
    %% Combine2 --> LossInterface23{Interface Loss <br/> L<sub>int₂₃</sub> <br/> (u<sub>total₂</sub>, u<sub>total₃</sub> on Γ₂₃)};
    %% Combine3 --> LossInterface23; %% Assuming Combine3 exists for Subdomain 3

    %% --- Total Loss and Optimization ---
    LossPDE1 --> TotalLoss[Total Loss Function <br/> L<sub>total</sub>];
    LossBC1 --> TotalLoss;
    LossPDE2 --> TotalLoss;
    LossBC2 --> TotalLoss;
    LossPDEN --> TotalLoss;
    LossBCN --> TotalLoss;
    LossInterface12 --> TotalLoss;
    %% LossInterface23 --> TotalLoss;

    TotalLoss --> Optimizer[Optimizer <br/> (e.g., Adam)];
    Optimizer -- Updates θ₁, θ₂, ..., θ<sub>N</sub> --> NN1;
    Optimizer --> NN2;
    Optimizer --> NNN;

    %% --- Styling (Optional, but helps readability) ---
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#lightgrey,stroke:#333,stroke-width:1px
    style Sub1 fill:#e6ffe6,stroke:#333,stroke-width:1px
    style Sub2 fill:#e6ffe6,stroke:#333,stroke-width:1px
    style SubN fill:#e6ffe6,stroke:#333,stroke-width:1px
    style NN1 fill:#ccf,stroke:#333,stroke-width:2px
    style NN2 fill:#ccf,stroke:#333,stroke-width:2px
    style NNN fill:#ccf,stroke:#333,stroke-width:2px
    style Combine1 fill:#cfc,stroke:#333,stroke-width:1px
    style Combine2 fill:#cfc,stroke:#333,stroke-width:1px
    style CombineN fill:#cfc,stroke:#333,stroke-width:1px
    style LossInterface12 fill:#fdb,stroke:#333,stroke-width:2px
    style TotalLoss fill:#ff9,stroke:#333,stroke-width:2px
    style Optimizer fill:#f96,stroke:#333,stroke-width:2px