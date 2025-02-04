import numpy as np


class Geim:
    """
    Implements the Generalized Empirical Interpolation Method (GEIM).
    
    This class processes a three-dimensional array of snapshots to compute GEIM bases and 
    reconstruct the solution space from measurement data.
    
    Attributes:
        __snaps (ndarray): A 3D array containing the snapshots with shape (Nx, Nfield, Nsnap), where:
                           - Nx: number of spatial points per field.
                           - Nfield: number of distinct fields.
                           - Nsnap: number of snapshots.
        __rank (int): The number of GEIM bases to compute (i.e., the reduced model rank).
        __Nx (int): The number of spatial points per field.
        __Nfield (int): The number of fields.
        __Nsnap (int): The number of snapshots.
        __Nx_stacked (int): The total number of rows in the stacked snapshot array, computed as Nx * Nfield.
        __norm_snaps (ndarray): A 2D array of shape (Nfield, Nsnap) containing the norm of each field for every snapshot.
        __stacked_snapshots (ndarray): A 2D array of shape (Nx_stacked, Nsnap) obtained by stacking the snapshot fields.
        __array_indexes_maximizing_position (ndarray): A 1D array of length 'rank' storing global indices in the stacked snapshot of sensor positions
                                                        that maximize the residual.
        indexes_position_sensors (ndarray): A 1D array of length 'rank' storing sensor position indices for each basis.
        matrix_holding_bases (ndarray): A 2D array of shape (Nx_stacked, rank) that holds the computed GEIM bases.
        A (ndarray): A square coefficient matrix of shape (rank, rank) used in reconstructing the solution.
        index_field_basis (ndarray): A 1D array of length 'rank' containing the field index for each computed basis.
        __J (ndarray): A 2D array of shape (Nx_stacked, Nsnap) representing the reconstructed solution space in stacked form.
        __reconstructed_solution (ndarray): The solution reconstructed by projecting measurement coefficients onto the GEIM bases.
    """
    def __init__(self, snaps, rank):
        """
        Initializes the Geim object with snapshot data and the desired GEIM rank.
        
        Parameters:
            snaps (array-like): A 3D array of snapshots with shape (Nx, Nfield, Nsnap).
            rank (int): The number of GEIM bases to compute.
            
        Raises:
            ValueError: If the provided snapshot array is not three-dimensional.
            TypeError: If the conversion of snapshots to a numpy array fails.
        """
        try:
            snaps_array = np.asarray(snaps).copy()
            if snaps_array.ndim != 3:
                raise ValueError("snaps must be a three-dimensional array")
            self.__snaps = snaps_array
            self.__rank = rank
            [self.__Nx, self.__Nfield, self.__Nsnap] = np.shape(self.__snaps)
            
            self.__Nx_stacked = int(self.__Nx * self.__Nfield)
            
            #generate norms
            self.__generate_norm_snaps()
            #stacks the fields on top of each other
            self.__stacker_fields()

            self.__array_indexes_maximizing_position = np.zeros((self.__rank), dtype=int)
            self.indexes_position_sensors = np.zeros((self.__rank), dtype=int)
            self.matrix_holding_bases = np.zeros((self.__Nx_stacked, self.__rank))

            self.A = np.zeros((self.__rank, self.__rank))

            self.index_field_basis = np.zeros(self.__rank, dtype=int)
            
            #this is the primary method that does all the main calculations
            self.__find_bases()

        except TypeError as e:
            print("Type error:", e)
        except ValueError as e:
            print("Value error:", e)
            
    def __stacker_fields(self):
        """
        Stacks the snapshot fields into a 2D array.
        
        Transposes the original snapshots from shape (Nx, Nfield, Nsnap) to (Nfield, Nx, Nsnap) 
        and then reshapes them into a 2D array of shape (Nx * Nfield, Nsnap).
        
        Raises:
            Exception: If an error occurs during the transposition or reshaping.
        """
        try:
            reshaped_snaps = np.transpose(self.__snaps, (1, 0, 2))
            self.__stacked_snapshots = reshaped_snaps.reshape((self.__Nx * self.__Nfield, self.__Nsnap))
        except Exception as e:
            print("An error occurred while stacking the snaps", e)
            raise
    
    def __generate_norm_snaps(self):
        """
        Generates the norm of each field for every snapshot.
        
        Computes the Euclidean norm (using np.linalg.norm) for the snapshot vectors of each field and 
        stores the result in a 2D array (__norm_snaps) of shape (Nfield, Nsnap).
        
        Raises:
            Exception: If an error occurs during the norm computation.
        """
        try:
            self.__norm_snaps = np.zeros((self.__Nfield, self.__Nsnap))
            for i in range(self.__Nfield):    
                for j in range(self.__Nsnap):
                    self.__norm_snaps[i, j] = np.linalg.norm(self.__snaps[:, i, j])
        except Exception as e:
            print("An error occurred while generating normalized snapshots:", e)
            raise
    
    def __generate_normalized_norm_residual_snaps(self, residual_snaps):
        """
        Computes normalized norm residual snapshots.
        
        For each field and snapshot, calculates the ratio of the norm of the residual snapshot to the 
        corresponding norm from the original snapshots.
        
        Parameters:
            residual_snaps (ndarray): A 2D array of residual snapshots with shape (Nx_stacked, Nsnap).
        
        Returns:
            ndarray: A copy of the normalized norm residual snapshots with shape (Nfield, Nsnap).
        
        Raises:
            Exception: If an error occurs during the computation.
        """
        try:
            normalized_norm_residual_snaps = np.zeros((self.__Nfield, self.__Nsnap))
            for j in range(self.__Nsnap):
                start_index = 0
                end_index = self.__Nx
                for i in range(self.__Nfield):    
                    norm_residual_snap = np.linalg.norm(residual_snaps[start_index: end_index, j])
                    normalized_norm_residual_snaps[i, j] = norm_residual_snap / self.__norm_snaps[i, j]
                    start_index = int((i + 1) * self.__Nx)
                    end_index = int((i + 2) * self.__Nx)
            return normalized_norm_residual_snaps.copy()
        except Exception as e:
            print("An error occurred while generating normalized norm of the residual of the snapshots:", e)
            raise
    
    def __coord_maximizing_snap_finder(self, residual_snaps):
        """
        Finds the coordinate of the snapshot that maximizes the relative residual error.
        
        Utilizes the normalized norm residual snapshots to identify the coordinate (field index, snapshot index)
        corresponding to the maximum error.
        
        Parameters:
            residual_snaps (ndarray): A 2D array of residual snapshots with shape (Nx_stacked, Nsnap).
        
        Returns:
            tuple: The (field index, snapshot index) where the residual is maximum.
        
        Raises:
            Exception: If an error occurs during the computation.
        """
        try:
            normalized_norm_residual_snaps = self.__generate_normalized_norm_residual_snaps(residual_snaps.copy())
            index_max = np.argmax(normalized_norm_residual_snaps)
            coord_max = np.unravel_index(index_max, normalized_norm_residual_snaps.shape)
            return coord_max
        except Exception as e:
            print("An error occurred while finding out the maximizing snap coordinate", e)
            raise
    
    def __reconstruct_solution_space(self, mat_A):
        """
        Reconstructs the solution space for each snapshot using the current GEIM bases.
        
        For each snapshot, extracts the measurement vector using indices from __array_indexes_maximizing_position,
        solves the linear system defined by mat_A to compute coefficients, and reconstructs the snapshot by projecting 
        these coefficients onto the GEIM bases.
        
        Parameters:
            mat_A (ndarray): The coefficient matrix of shape (n_rows, n_rows) used in the linear solve.
        
        Raises:
            Exception: If an error occurs during the reconstruction process.
        """
        try:    
            [n_rows, n_cols] = mat_A.shape
            self.__J = np.zeros((self.__Nx_stacked, self.__Nsnap))
            for snapshot_i in range(self.__Nsnap):
                b = np.asarray(self.__stacked_snapshots[self.__array_indexes_maximizing_position[:n_rows], snapshot_i])
                d_i = np.linalg.solve(mat_A, b)
                self.__J[:, snapshot_i] = self.matrix_holding_bases[:, :n_rows] @ d_i
        except Exception as e:
            print("An error occurred while reconstructing the solution space", e)
            raise
    
    def __find_bases(self):
        """
        Iteratively finds and constructs the GEIM bases.
        
        Initially, identifies the snapshot with the maximum norm to set the first basis. Then, for each subsequent
        basis, computes the residual, locates the coordinate with the maximum relative error, updates the basis matrix,
        sensor index arrays, and the coefficient matrix A accordingly.
        
        Raises:
            Exception: If an error occurs during the basis-finding process.
        """
        try:
            # For the first basis:
            index_max = np.argmax(self.__norm_snaps)
            coord_max = np.unravel_index(index_max, self.__norm_snaps.shape)
            self.matrix_holding_bases[:, 0] = self.__stacked_snapshots[:, coord_max[1]].copy()
    
            index_basis = 0
            i = coord_max[0]
            index_maximizing_position = np.argmax(np.abs(self.matrix_holding_bases[i * self.__Nx:(i + 1) * self.__Nx, index_basis]))
            self.__array_indexes_maximizing_position[index_basis] = index_maximizing_position + i * self.__Nx
            self.indexes_position_sensors[index_basis] = index_maximizing_position
            self.index_field_basis[index_basis] = i
            self.A[0, 0] = self.matrix_holding_bases[self.__array_indexes_maximizing_position[index_basis], index_basis]
            self.__reconstruct_solution_space(self.A[:index_basis + 1, :index_basis + 1])
    
            for index_basis in range(1, self.__rank):
                residual_snaps = self.__stacked_snapshots - self.__J
                coord_max = self.__coord_maximizing_snap_finder(residual_snaps)
                self.matrix_holding_bases[:, index_basis] = residual_snaps[:, coord_max[1]].copy()
                i = coord_max[0]
                index_maximizing_position = np.argmax(np.abs(self.matrix_holding_bases[i * self.__Nx:(i + 1) * self.__Nx, index_basis]))
                self.__array_indexes_maximizing_position[index_basis] = index_maximizing_position + i * self.__Nx
                self.index_field_basis[index_basis] = i
                self.indexes_position_sensors[index_basis] = index_maximizing_position
    
                for i in range(index_basis + 1):
                    for j in range(index_basis + 1):
                        self.A[i, j] = self.matrix_holding_bases[self.__array_indexes_maximizing_position[i], j].copy()
                
                self.__reconstruct_solution_space(self.A[:index_basis + 1, :index_basis + 1])
        except Exception as e:
            print("An error occurred while finding out the bases.", e)
            raise
    
    def reconstruct_solution(self, measurments):
        """
        Reconstructs the solution from the given measurements.
        
        Solves the linear system defined by the coefficient matrix A and the measurement vector to obtain the 
        coefficients, then reconstructs the solution by using these coefficients onto the GEIM bases.
        Finally, unpacks the reconstructed solution into its original shape.
        
        Parameters:
            measurments (ndarray): The measurement vector used for reconstructing the solution.
        
        Returns:
            ndarray: The reconstructed solution in its original shape (Nx, Nfield).
        
        Raises:
            Exception: If an error occurs during the reconstruction process.
        """
        try:
            coeffs = np.linalg.solve(self.A, measurments)
            self.__reconstructed_solution = self.matrix_holding_bases @ coeffs
            fields_in_column_solution = self.__unstacker_fields()
            return fields_in_column_solution
        except Exception as e:
            print("An error occurred while reconstructing the solution:", e)
            raise 
    
    def __unstacker_fields(self):
        """
        Unstacks the reconstructed solution from its 2D stacked form to its original 2D shape.
        
        Reshapes the reconstructed solution (which is in a stacked form of shape (Nfield, Nx)) back to its original
        orientation (Nx, Nfield) by transposing.
        
        Returns:
            ndarray: The unstacked solution in its original shape (Nx, Nfield).
        
        Raises:
            Exception: If an error occurs during the unstacking process.
        """
        try:
            unstacked_snap = self.__reconstructed_solution.reshape((self.__Nfield, self.__Nx))
            original_shaped_snap = np.transpose(unstacked_snap, (1, 0))
            return original_shaped_snap
        except Exception as e:
            print("An error occurred while unstacking the snaps", e)
            raise
