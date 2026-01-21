export interface ProcessingResult {
  originalImage: string;
  processedImage: string;
  segmentedImage: string;
  footprintLength: number;
  areaA: number;
  areaB: number;
  areaC: number;
  totalArea: number;
  archIndex: number;
  classification: 'flat' | 'normal';
  processingSteps: string[];
}

// Convert image to grayscale
function toGrayscale(imageData: ImageData): ImageData {
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const avg = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
    data[i] = avg;
    data[i + 1] = avg;
    data[i + 2] = avg;
  }
  return imageData;
}

// Enhance contrast using histogram stretching
function enhanceContrast(imageData: ImageData): ImageData {
  const data = imageData.data;
  let min = 255, max = 0;
  
  for (let i = 0; i < data.length; i += 4) {
    const val = data[i];
    if (val < min) min = val;
    if (val > max) max = val;
  }
  
  const range = max - min || 1;
  
  for (let i = 0; i < data.length; i += 4) {
    const normalized = ((data[i] - min) / range) * 255;
    data[i] = normalized;
    data[i + 1] = normalized;
    data[i + 2] = normalized;
  }
  
  return imageData;
}

// Create binary mask using adaptive thresholding
// Returns a 2D boolean array where true = footprint pixel, false = background
function createBinaryMask(imageData: ImageData, blockSize: number = 25, c: number = 10): boolean[][] {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  
  // Create 2D boolean mask
  const mask: boolean[][] = Array.from({ length: height }, () => Array(width).fill(false));
  
  const halfBlock = Math.floor(blockSize / 2);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let count = 0;
      
      // Calculate local mean in block
      for (let dy = -halfBlock; dy <= halfBlock; dy++) {
        for (let dx = -halfBlock; dx <= halfBlock; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const idx = (ny * width + nx) * 4;
            sum += data[idx];
            count++;
          }
        }
      }
      
      const mean = sum / count;
      const idx = (y * width + x) * 4;
      const threshold = mean - c;
      
      // Dark pixels (footprint) = true, light pixels (background) = false
      mask[y][x] = data[idx] < threshold;
    }
  }
  
  return mask;
}

// Morphological closing to clean up the mask
function morphologicalClose(mask: boolean[][], kernelSize: number = 5): boolean[][] {
  const height = mask.length;
  const width = mask[0].length;
  const half = Math.floor(kernelSize / 2);
  
  // Dilation
  const dilated: boolean[][] = Array.from({ length: height }, () => Array(width).fill(false));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let hasTrue = false;
      for (let dy = -half; dy <= half && !hasTrue; dy++) {
        for (let dx = -half; dx <= half && !hasTrue; dx++) {
          const ny = Math.min(Math.max(y + dy, 0), height - 1);
          const nx = Math.min(Math.max(x + dx, 0), width - 1);
          if (mask[ny][nx]) hasTrue = true;
        }
      }
      dilated[y][x] = hasTrue;
    }
  }
  
  // Erosion
  const eroded: boolean[][] = Array.from({ length: height }, () => Array(width).fill(false));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let allTrue = true;
      for (let dy = -half; dy <= half && allTrue; dy++) {
        for (let dx = -half; dx <= half && allTrue; dx++) {
          const ny = Math.min(Math.max(y + dy, 0), height - 1);
          const nx = Math.min(Math.max(x + dx, 0), width - 1);
          if (!dilated[ny][nx]) allTrue = false;
        }
      }
      eroded[y][x] = allTrue;
    }
  }
  
  return eroded;
}

// Find connected components and return mask of the largest one
function extractLargestComponent(mask: boolean[][]): boolean[][] {
  const height = mask.length;
  const width = mask[0].length;
  const visited: boolean[][] = Array.from({ length: height }, () => Array(width).fill(false));
  const labels: number[][] = Array.from({ length: height }, () => Array(width).fill(0));
  
  let currentLabel = 0;
  const componentSizes: Map<number, number> = new Map();
  
  // Flood fill to find connected components
  for (let startY = 0; startY < height; startY++) {
    for (let startX = 0; startX < width; startX++) {
      if (mask[startY][startX] && !visited[startY][startX]) {
        currentLabel++;
        const queue: [number, number][] = [[startX, startY]];
        let size = 0;
        
        while (queue.length > 0) {
          const [x, y] = queue.shift()!;
          
          if (x < 0 || x >= width || y < 0 || y >= height) continue;
          if (visited[y][x] || !mask[y][x]) continue;
          
          visited[y][x] = true;
          labels[y][x] = currentLabel;
          size++;
          
          queue.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
        }
        
        componentSizes.set(currentLabel, size);
      }
    }
  }
  
  // Find largest component label
  let largestLabel = 0;
  let largestSize = 0;
  componentSizes.forEach((size, label) => {
    if (size > largestSize) {
      largestSize = size;
      largestLabel = label;
    }
  });
  
  // Create mask containing only the largest component
  const result: boolean[][] = Array.from({ length: height }, () => Array(width).fill(false));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      result[y][x] = labels[y][x] === largestLabel;
    }
  }
  
  return result;
}

// Get bounding box of the mask
function getMaskBounds(mask: boolean[][]): { minX: number; maxX: number; minY: number; maxY: number } {
  const height = mask.length;
  const width = mask[0].length;
  let minX = width, maxX = 0, minY = height, maxY = 0;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y][x]) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }
  
  return { minX, maxX, minY, maxY };
}

// Remove toe regions from the mask
// Toes are typically the top portion with scattered/separate regions
function removeToeRegion(mask: boolean[][]): boolean[][] {
  const height = mask.length;
  const width = mask[0].length;
  const bounds = getMaskBounds(mask);
  
  const footprintHeight = bounds.maxY - bounds.minY;
  // Remove top 18% which typically contains toe impressions
  const toeRemovalLine = bounds.minY + Math.floor(footprintHeight * 0.18);
  
  // Create new mask with toe region removed
  const result: boolean[][] = Array.from({ length: height }, () => Array(width).fill(false));
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Only keep pixels below the toe removal line
      result[y][x] = y >= toeRemovalLine && mask[y][x];
    }
  }
  
  return result;
}

// Calculate areas for three equal vertical sections using ONLY mask pixels
// This is the critical function - we count actual footprint pixels, not rectangles
function calculateMaskBasedAreas(mask: boolean[][]): {
  areaA: number;
  areaB: number;
  areaC: number;
  totalArea: number;
  footprintLength: number;
  regionBoundaries: { y1: number; y2: number };
  actualBounds: { minY: number; maxY: number };
} {
  const height = mask.length;
  const width = mask[0].length;
  
  // Step 1: Find actual footprint bounds from the mask
  let actualMinY = height;
  let actualMaxY = 0;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y][x]) {
        actualMinY = Math.min(actualMinY, y);
        actualMaxY = Math.max(actualMaxY, y);
      }
    }
  }
  
  // Step 2: Calculate footprint length (vertical span of mask pixels only)
  const footprintLength = actualMaxY - actualMinY;
  
  if (footprintLength <= 0) {
    return {
      areaA: 0,
      areaB: 0,
      areaC: 0,
      totalArea: 0,
      footprintLength: 0,
      regionBoundaries: { y1: 0, y2: 0 },
      actualBounds: { minY: 0, maxY: 0 }
    };
  }
  
  // Step 3: Divide into three EQUAL vertical sections
  const sectionHeight = footprintLength / 3;
  
  // y1: boundary between Area C (forefoot/top) and Area B (midfoot/middle)
  // y2: boundary between Area B (midfoot/middle) and Area A (rearfoot/bottom)
  const y1 = actualMinY + sectionHeight;
  const y2 = actualMinY + 2 * sectionHeight;
  
  // Step 4: Count ONLY actual footprint mask pixels in each section
  // This is mask-based counting, NOT rectangular area calculation
  let areaA = 0; // Rearfoot (bottom third)
  let areaB = 0; // Midfoot (middle third)
  let areaC = 0; // Forefoot (top third)
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // CRITICAL: Only count if this pixel is part of the footprint mask
      if (mask[y][x]) {
        if (y >= y2) {
          // This footprint pixel is in the rearfoot region
          areaA++;
        } else if (y >= y1) {
          // This footprint pixel is in the midfoot region
          areaB++;
        } else if (y >= actualMinY) {
          // This footprint pixel is in the forefoot region
          areaC++;
        }
      }
      // Background pixels (mask[y][x] === false) are NEVER counted
    }
  }
  
  const totalArea = areaA + areaB + areaC;
  
  return {
    areaA,
    areaB,
    areaC,
    totalArea,
    footprintLength,
    regionBoundaries: { y1, y2 },
    actualBounds: { minY: actualMinY, maxY: actualMaxY }
  };
}

// Create processed image showing footprint mask with green background
function createProcessedImage(
  originalData: ImageData,
  mask: boolean[][],
  width: number,
  height: number
): ImageData {
  const result = new ImageData(width, height);
  const data = result.data;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      
      if (mask[y][x]) {
        // Footprint pixel - show in dark gray to represent the silhouette
        const origIdx = (y * width + x) * 4;
        const gray = Math.min(
          80,
          (originalData.data[origIdx] * 0.299 + 
           originalData.data[origIdx + 1] * 0.587 + 
           originalData.data[origIdx + 2] * 0.114) * 0.5
        );
        data[idx] = gray;
        data[idx + 1] = gray;
        data[idx + 2] = gray;
        data[idx + 3] = 255;
      } else {
        // Background - show as green (removed/excluded region)
        data[idx] = 76;
        data[idx + 1] = 175;
        data[idx + 2] = 80;
        data[idx + 3] = 255;
      }
    }
  }
  
  return result;
}

// Create segmented image showing the three regions following the ACTUAL footprint contour
// This creates a silhouette-like visualization, NOT rectangular regions
function createSegmentedImage(
  mask: boolean[][],
  width: number,
  height: number,
  regionBoundaries: { y1: number; y2: number },
  actualBounds: { minY: number; maxY: number }
): ImageData {
  const result = new ImageData(width, height);
  const data = result.data;
  
  // Colors for each region
  const colorA = { r: 233, g: 91, b: 133 };  // Rearfoot - rose/pink
  const colorB = { r: 251, g: 189, b: 35 };  // Midfoot - amber/gold
  const colorC = { r: 66, g: 133, b: 244 };  // Forefoot - blue
  const bgColor = { r: 76, g: 175, b: 80 };  // Green background
  
  const { y1, y2 } = regionBoundaries;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      
      // CRITICAL: Only color pixels that are part of the footprint mask
      // This ensures the segmented image follows the actual footprint contour
      if (mask[y][x]) {
        let color;
        
        if (y >= y2) {
          // Rearfoot region (bottom third of footprint)
          color = colorA;
        } else if (y >= y1) {
          // Midfoot region (middle third of footprint)
          color = colorB;
        } else {
          // Forefoot region (top third of footprint, after toe removal)
          color = colorC;
        }
        
        data[idx] = color.r;
        data[idx + 1] = color.g;
        data[idx + 2] = color.b;
        data[idx + 3] = 255;
      } else {
        // Background pixel - NOT part of footprint, show as green
        data[idx] = bgColor.r;
        data[idx + 1] = bgColor.g;
        data[idx + 2] = bgColor.b;
        data[idx + 3] = 255;
      }
    }
  }
  
  // Draw horizontal division lines at the section boundaries
  // Lines are drawn across the footprint width only
  const lineColor = { r: 255, g: 255, b: 255 };
  const lineThickness = 2;
  
  // Find the horizontal extent of the footprint at each boundary
  const drawLineAtY = (targetY: number) => {
    const yInt = Math.floor(targetY);
    if (yInt < 0 || yInt >= height) return;
    
    // Find leftmost and rightmost footprint pixels near this y
    let leftX = width, rightX = 0;
    for (let dy = -3; dy <= 3; dy++) {
      const checkY = yInt + dy;
      if (checkY >= 0 && checkY < height) {
        for (let x = 0; x < width; x++) {
          if (mask[checkY][x]) {
            leftX = Math.min(leftX, x);
            rightX = Math.max(rightX, x);
          }
        }
      }
    }
    
    // Draw line only within the footprint bounds
    if (leftX < rightX) {
      for (let t = -lineThickness / 2; t <= lineThickness / 2; t++) {
        const drawY = yInt + Math.floor(t);
        if (drawY >= 0 && drawY < height) {
          for (let x = leftX; x <= rightX; x++) {
            // Only draw on footprint pixels
            if (mask[drawY][x] || (drawY >= 0 && drawY < height && mask[Math.max(0, drawY - 1)][x]) || 
                (drawY < height - 1 && mask[drawY + 1][x])) {
              const idx = (drawY * width + x) * 4;
              data[idx] = lineColor.r;
              data[idx + 1] = lineColor.g;
              data[idx + 2] = lineColor.b;
            }
          }
        }
      }
    }
  };
  
  drawLineAtY(y1);
  drawLineAtY(y2);
  
  return result;
}

export async function processFootprintImage(imageFile: File): Promise<ProcessingResult> {
  const processingSteps: string[] = [];
  
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(imageFile);
    
    img.onload = () => {
      try {
        const width = img.width;
        const height = img.height;
        
        processingSteps.push(`Image loaded: ${width} × ${height} pixels`);
        
        // Create canvas for processing
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d')!;
        
        // Draw original image
        ctx.drawImage(img, 0, 0);
        const originalImageData = ctx.getImageData(0, 0, width, height);
        const originalDataUrl = canvas.toDataURL('image/png');
        
        // Create working copy for grayscale processing
        const workingImageData = ctx.getImageData(0, 0, width, height);
        
        // Step 1: Convert to grayscale
        processingSteps.push('Converting to grayscale...');
        toGrayscale(workingImageData);
        
        // Step 2: Enhance contrast
        processingSteps.push('Enhancing contrast...');
        enhanceContrast(workingImageData);
        
        // Step 3: Create binary footprint mask using adaptive thresholding
        processingSteps.push('Creating binary footprint mask...');
        let footprintMask = createBinaryMask(workingImageData, 25, 10);
        
        // Step 4: Morphological closing to clean up the mask
        processingSteps.push('Applying morphological cleanup...');
        footprintMask = morphologicalClose(footprintMask, 5);
        
        // Step 5: Extract the largest connected component (main footprint body)
        processingSteps.push('Extracting largest connected component...');
        footprintMask = extractLargestComponent(footprintMask);
        
        // Step 6: Remove toe region from the mask
        processingSteps.push('Removing toe region from mask...');
        footprintMask = removeToeRegion(footprintMask);
        
        // Step 7: Calculate mask-based areas
        processingSteps.push('Calculating mask-based region areas...');
        const areaResult = calculateMaskBasedAreas(footprintMask);
        
        processingSteps.push(`Footprint length (from mask): ${areaResult.footprintLength} px`);
        processingSteps.push(`Area A (rearfoot mask pixels): ${areaResult.areaA.toLocaleString()} px`);
        processingSteps.push(`Area B (midfoot mask pixels): ${areaResult.areaB.toLocaleString()} px`);
        processingSteps.push(`Area C (forefoot mask pixels): ${areaResult.areaC.toLocaleString()} px`);
        processingSteps.push(`Total footprint pixels: ${areaResult.totalArea.toLocaleString()} px`);
        
        // Step 8: Calculate Arch Index
        const archIndex = areaResult.totalArea > 0 
          ? areaResult.areaB / areaResult.totalArea 
          : 0;
        processingSteps.push(`Arch Index = B / (A + B + C) = ${archIndex.toFixed(4)}`);
        
        // Step 9: Classification based on threshold
        const classification: 'flat' | 'normal' = archIndex > 0.28 ? 'flat' : 'normal';
        processingSteps.push(`Classification: ${classification === 'flat' ? 'Tendency toward flat foot (AI > 0.2800)' : 'Normal foot arch (AI ≤ 0.2800)'}`);
        
        // Create visualization images
        // Processed image: shows footprint silhouette with green background
        const processedImageData = createProcessedImage(
          originalImageData,
          footprintMask,
          width,
          height
        );
        ctx.putImageData(processedImageData, 0, 0);
        const processedDataUrl = canvas.toDataURL('image/png');
        
        // Segmented image: shows three colored regions following footprint contour
        const segmentedImageData = createSegmentedImage(
          footprintMask,
          width,
          height,
          areaResult.regionBoundaries,
          areaResult.actualBounds
        );
        ctx.putImageData(segmentedImageData, 0, 0);
        const segmentedDataUrl = canvas.toDataURL('image/png');
        
        URL.revokeObjectURL(url);
        
        resolve({
          originalImage: originalDataUrl,
          processedImage: processedDataUrl,
          segmentedImage: segmentedDataUrl,
          footprintLength: areaResult.footprintLength,
          areaA: areaResult.areaA,
          areaB: areaResult.areaB,
          areaC: areaResult.areaC,
          totalArea: areaResult.totalArea,
          archIndex,
          classification,
          processingSteps
        });
        
      } catch (error) {
        URL.revokeObjectURL(url);
        reject(error);
      }
    };
    
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load image'));
    };
    
    img.src = url;
  });
}
