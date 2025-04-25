function u_k= myca_svd(m_desired)

    B_matrix = 89*single([-0.0058776699,-0.0058776699,-0.0058776699,-0.0058776699,-0.0058776699,-0.0058776699; ...
                      -0.076463461,-0.15292691,-0.076463461,0.076463461,0.15292691,0.076463461; ...
                       0.12950730,0,-0.12950730,-0.12950730,0,0.12950730; ...
                      -0.0085608540,0.0085608540,-0.0085608540,0.0085608540,-0.0085608540,0.0085608540]);
    
    [n, m] = size(B_matrix); 
    
    [U, B_bidiag, V] = bidiagonalize_golub_kahan(B_matrix);
    
    [U_bidiag, Sigma, V_bidiag] = bidiagonalSVD(B_bidiag);
    Sigma_diag = diag(Sigma);    
    U_updated = U * U_bidiag;
    V_updated = V * V_bidiag;    

    tol = 1e-6;
    
    Sigma_inv = zeros(m, n); 
    for i = 1:min(n, m)
        if Sigma_diag(i) > tol
            Sigma_inv(i, i) = 1 / Sigma_diag(i); % Invert singular value
        end
    end

    B_pseudo_inv = V_updated * Sigma_inv * U_updated';    
    u_k= B_pseudo_inv * m_desired;

end

function [U, Sigma, V] = bidiagonalSVD(B)

    [n, m] = size(B);
    U = eye(n); 
    V = eye(m); 
    tol = 1e-10; 
    maxIter = 1000; 

    
    for iter = 1:maxIter        
        offDiagonalNorm = sqrt(sum(sum(B.^2)) - sum(diag(B).^2));
        if offDiagonalNorm < tol
            break;
        end

        for i = 1:min(n, m) - 1
            
            [c, s] = givensRotation(B(i, i), B(i + 1, i));

            
            G = [c, s; -s, c];
            B(i:i+1, :) = G' * B(i:i+1, :);
            U(:, i:i+1) = U(:, i:i+1) * G;

            
            if i < m
                [c, s] = givensRotation(B(i, i), B(i, i + 1));
                G = [c, s; -s, c];
                B(:, i:i+1) = B(:, i:i+1) * G;
                V(:, i:i+1) = V(:, i:i+1) * G;
            end
        end
    end

    
    Sigma = zeros(n, m);
    for i = 1:min(n, m)
        Sigma(i, i) = abs(B(i, i));
        if B(i, i) < 0            
            U(:, i) = -U(:, i);
        end
    end
end

function [c, s] = givensRotation(a, b)    
    if b == 0
        c = single(1);
        s = single(0);
    else
        if abs(b) > abs(a)
            t = single(-a / b);
            s = single(1 / sqrt(1 + t^2));
            c = single(s * t);
        else
            t = single(-b / a);
            c = single(1 / sqrt(1 + t^2));
            s = single(c * t);
        end
    end
end

function [U, B_bidiag, V] = bidiagonalize_golub_kahan(B)
    [n, m] = size(B);
    U = single(eye(n)); 
    V = single(eye(m)); 
    B_bidiag = B;
    
    for k = 1:min(n, m)
        x = B_bidiag(k:end, k); 
        e1 = zeros(length(x), 1, 'single');
        e1(1) = single(norm(x));
        v = x - e1; 
        if norm(v) < single(1e-12) 
            v = zeros(size(v), 'single'); 
        else
            v = single(v / norm(v)); 
        end

        Hk = single(eye(n)); 
        Hk(k:end, k:end) = single(eye(length(x)) - 2 * (v * v'));
        B_bidiag = Hk * B_bidiag; 
        U = U * Hk; 

        if k <= m - 1
            x = B_bidiag(k, k+1:end)'; 
            e1 = zeros(length(x), 1, 'single');
            e1(1) = single(norm(x));
            v = x - e1; 
            if norm(v) < single(1e-12) 
                v = zeros(size(v), 'single'); 
            else
                v = single(v / norm(v)); 
            end

            Hk = single(eye(m)); 
            Hk(k+1:end, k+1:end) = eye(length(x)) - 2 * (v * v');
            B_bidiag = B_bidiag * Hk; 
            V = V * Hk; 
        end
    end
end
