#include "TSDFVoxelGrid.h"
#include <cmath>

VoxelGrid::VoxelGrid(unsigned int spatialExtendHorizontally, unsigned int spatialExtendVertically, unsigned int voxelsPerUnit) : m_spatialExtendHorizontally(spatialExtendHorizontally),
																																 m_spatialExtendVertically(spatialExtendVertically), m_voxelsPerUnit(voxelsPerUnit)
{
	unsigned long long horizontal = std::pow(spatialExtendHorizontally * voxelsPerUnit, 2);
	unsigned long long numberVoxels = std::floor( horizontal * (spatialExtendVertically * voxelsPerUnit));
	m_voxelGrid = std::vector<Vector2f>(numberVoxels, Vector2f(0.0, 0.0));
	Vector2f test = m_voxelGrid.at(0);
}