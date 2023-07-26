#include "PointCloudToMesh.h"

bool valid(Vector3f v)
{
	// check whether a point is valid
	return (v.x() != MINF);
}

float distance(Vector3f a, Vector3f b)
{
	return sqrt((a - b).transpose() * (a - b));
}

bool writeMesh(Vector3f *vertices, unsigned int width, unsigned int height, const std::string &filename)
{
	float edgeThreshold = 0.015f;
	unsigned int nVertices = width * height;
	std::vector<int> triangles_to_lower_right;
	std::vector<int> triangles_to_upper_left;

	size_t i = 0;
	for (; i < (width * height); i++)
	{
		int x = i % width;
		int y = std::floor(i / width);
		//					  |"'/
		// to the lower right |/
		if (x + 1 < width && y + 1 < height)
		{
			if (valid(vertices[i]) && valid(vertices[i + 1]) && valid(vertices[i + width]))
			{
				// check if distance is still within threshold
				if (distance(vertices[i], vertices[i + 1]) < edgeThreshold &&
					distance(vertices[i], vertices[i + width]) < edgeThreshold &&
					distance(vertices[i + 1], vertices[i + width]) < edgeThreshold)
				{
					triangles_to_lower_right.push_back(i);
				}
			} 
		}
		
		//					   /|
		// to the upper left  /_|
		if (x - 1 >= 0 && y - 1 >= 0)
		{
			if (valid(vertices[i]) && valid(vertices[i - 1]) && valid(vertices[i - width]))
			{
				// check if distance is still  within threshold
				if (distance(vertices[i], vertices[i - 1]) < edgeThreshold &&
					distance(vertices[i], vertices[i - width]) < edgeThreshold &&
					distance(vertices[i + 1], vertices[i - width]) < edgeThreshold)
				{
					triangles_to_upper_left.push_back(i);
				}
			}
		}
	}
	// Determine number of valid faces
	unsigned nFaces = triangles_to_lower_right.size() + triangles_to_upper_left.size();

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open())
		return false;
	// write header
	outFile << "COFF" << std::endl;

	outFile << "# numVertices numFaces numEdges" << std::endl;

	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices
	outFile << "# list of vertices" << std::endl;
	outFile << "# X Y Z" << std::endl;
	for (size_t i = 0; i < nVertices; i++)
	{
		if (valid(vertices[i]))
		{
			outFile << std::to_string(vertices[i].x()) + " ";
			outFile << std::to_string(vertices[i].y()) + " ";
			outFile << std::to_string(vertices[i].z()) + " ";
		}
		else
		{
			outFile << "0 0 0 ";
		}
		outFile << std::endl;
	}

	// TODO: save valid faces
	outFile << "# list of faces" << std::endl;
	outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;
	for (size_t i = 0; i < triangles_to_lower_right.size(); i++)
	{
		int valid_idx = triangles_to_lower_right[i];
		outFile << "3 " + std::to_string(valid_idx) + " " + std::to_string(valid_idx + width) + " " + std::to_string(valid_idx + 1) + "\n";
	}
	for (size_t i = 0; i < triangles_to_upper_left.size(); i++) {

		int valid_idx = triangles_to_upper_left[i];
		outFile << "3 " + std::to_string(valid_idx -1) + " " + std::to_string(valid_idx) + " " + std::to_string(valid_idx - width) + "\n";

	}
	// close file
	outFile.close();

	return true;
}